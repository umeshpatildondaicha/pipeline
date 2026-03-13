// ═══════════════════════════════════════════════════════════════════
// NocInferenceService.java
// Spring Boot service that loads ONNX models and runs predictions.
// Watches for new model files → auto hot-swap with zero downtime.
// ═══════════════════════════════════════════════════════════════════

package com.noc.ml.service;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

@Slf4j
@Service
public class NocInferenceService {

    /**
     * Directory where Python pipeline writes trained ONNX models and manifest.json.
     * Must be set to a real path in application.properties / environment.
     * There is intentionally no absolute-path default — the application will fail
     * fast at startup rather than silently looking in a non-existent location.
     *
     * Example application.properties entry:
     *   noc.models.deploy-dir=${user.home}/noc-pipeline/data/models/deploy
     */
    @Value("${noc.models.deploy-dir}")
    private String deployDir;

    /**
     * How often (ms) the watcher checks manifest.json for a new model version.
     * Default: 10 seconds.  Override via noc.models.watcher-interval-ms.
     */
    @Value("${noc.models.watcher-interval-ms:10000}")
    private long watcherIntervalMs;

    /**
     * Minimum confidence threshold for propagation rule predictions.
     * Rules with confidence below this value are filtered out.
     * Default: 0.3.  Override via noc.models.min-propagation-confidence.
     */
    @Value("${noc.models.min-propagation-confidence:0.3}")
    private double minPropagationConfidence;

    // ── Model sessions (atomic references for hot-swap)
    private final AtomicReference<OrtSession> rootCauseSession   = new AtomicReference<>();
    private final AtomicReference<OrtSession> sequenceSession    = new AtomicReference<>();
    private final AtomicReference<OrtSession> anomalySession      = new AtomicReference<>();
    private final AtomicReference<Map<String, Object>> propagationRules = new AtomicReference<>();

    private OrtEnvironment env;
    private List<String>   rootCauseFeatures;
    private List<String>   anomalyFeatures;
    private List<String>   rootCauseLabels;
    private Map<String, Integer> sequenceVocab;  // alarm_code -> index (0=pad, 1=unk)
    private List<String>   sequenceLabels;
    private static final int MAX_SEQ_LEN = 20;
    private long           currentModelVersion = -1;

    private final ObjectMapper mapper = new ObjectMapper();

    // ─────────────────────────────────────────────────────────────
    // INITIALIZATION
    // ─────────────────────────────────────────────────────────────
    @PostConstruct
    public void init() throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();
        loadAllModels();
        startModelWatcher();
    }

    @PreDestroy
    public void destroy() throws OrtException {
        closeSession(rootCauseSession.get());
        closeSession(sequenceSession.get());
        closeSession(anomalySession.get());
        if (env != null) env.close();
    }

    // ─────────────────────────────────────────────────────────────
    // LOAD ALL MODELS
    // ─────────────────────────────────────────────────────────────
    private void loadAllModels() throws OrtException, IOException {
        log.info("Loading NOC ML models from: {}", deployDir);

        // Root Cause Classifier
        Path rcPath = Paths.get(deployDir, "root_cause_classifier.onnx");
        if (Files.exists(rcPath)) {
            OrtSession newSession = env.createSession(
                rcPath.toString(),
                new OrtSession.SessionOptions()
            );
            closeSession(rootCauseSession.getAndSet(newSession));
            log.info("  ✅ Root Cause Classifier loaded");
        }

        // KPI Anomaly Detector
        Path anomalyPath = Paths.get(deployDir, "kpi_anomaly_detector.onnx");
        if (Files.exists(anomalyPath)) {
            OrtSession newSession = env.createSession(
                anomalyPath.toString(),
                new OrtSession.SessionOptions()
            );
            closeSession(anomalySession.getAndSet(newSession));
            log.info("  ✅ KPI Anomaly Detector loaded");
        }

        // Sequence model (LSTM on alarm sequence → root cause)
        Path seqPath = Paths.get(deployDir, "root_cause_sequence.onnx");
        if (Files.exists(seqPath)) {
            OrtSession newSession = env.createSession(
                seqPath.toString(),
                new OrtSession.SessionOptions()
            );
            closeSession(sequenceSession.getAndSet(newSession));
            log.info("  ✅ Root Cause Sequence model loaded");
        }
        Path seqVocabPath = Paths.get(deployDir, "sequence_model_vocab.json");
        if (Files.exists(seqVocabPath)) {
            sequenceVocab = mapper.readValue(seqVocabPath.toFile(),
                mapper.getTypeFactory().constructMapType(Map.class, String.class, Integer.class));
        }
        sequenceLabels = loadFeatureList("sequence_model_labels.json");
        if (sequenceLabels.isEmpty()) sequenceLabels = rootCauseLabels;

        // Alarm Propagation Rules (JSON lookup — no ONNX needed)
        Path rulesPath = Paths.get(deployDir, "alarm_propagation_rules.json");
        if (Files.exists(rulesPath)) {
            Map<String, Object> rules = mapper.readValue(
                rulesPath.toFile(),
                mapper.getTypeFactory().constructMapType(
                    Map.class, String.class, Object.class
                )
            );
            propagationRules.set(rules);
            log.info("  ✅ Propagation Rules loaded ({} root alarms)", rules.size());
        }

        // Feature column lists
        rootCauseFeatures = loadFeatureList("root_cause_features.json");
        anomalyFeatures   = loadFeatureList("kpi_anomaly_features.json");
        rootCauseLabels   = loadFeatureList("root_cause_labels.json");

        // Read manifest version
        Path manifestPath = Paths.get(deployDir, "manifest.json");
        if (Files.exists(manifestPath)) {
            Map<?, ?> manifest = mapper.readValue(manifestPath.toFile(), Map.class);
            currentModelVersion = ((Number) manifest.get("version")).longValue();
            log.info("  Model version: v{}", currentModelVersion);
        }
    }

    @SuppressWarnings("unchecked")
    private List<String> loadFeatureList(String filename) throws IOException {
        Path p = Paths.get(deployDir, filename);
        if (!Files.exists(p)) return Collections.emptyList();
        return mapper.readValue(p.toFile(), List.class);
    }

    // ─────────────────────────────────────────────────────────────
    // MODEL WATCHER — detects new ONNX files every 10 seconds
    // ─────────────────────────────────────────────────────────────
    @Scheduled(fixedDelayString = "${noc.models.watcher-interval-ms:10000}")
    public void checkForModelUpdates() {
        try {
            Path manifestPath = Paths.get(deployDir, "manifest.json");
            if (!Files.exists(manifestPath)) return;

            Map<?, ?> manifest = mapper.readValue(manifestPath.toFile(), Map.class);
            long newVersion = ((Number) manifest.get("version")).longValue();

            if (newVersion > currentModelVersion) {
                log.info("New model version detected: v{} → v{}",
                          currentModelVersion, newVersion);
                loadAllModels();
                log.info("✅ Models hot-swapped to v{}", newVersion);
            }
        } catch (Exception e) {
            log.warn("Model watcher error: {}", e.getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────
    // ROOT CAUSE PREDICTION
    // ─────────────────────────────────────────────────────────────
    public RootCausePrediction predictRootCause(AlarmContext context) {
        OrtSession session = rootCauseSession.get();
        if (session == null) {
            return RootCausePrediction.unknown("Model not loaded");
        }

        try {
            // Build feature vector from alarm context
            float[] features = buildRootCauseFeatures(context);

            // Run ONNX inference
            long[] shape   = {1L, (long) features.length};
            OnnxTensor input = OnnxTensor.createTensor(env, features, shape);

            Map<String, OnnxTensor> inputMap = Map.of(
                session.getInputNames().iterator().next(), input
            );

            OrtSession.Result result = session.run(inputMap);

            // Extract prediction
            long   classIdx     = ((long[]) result.get(0).getValue())[0];
            float  confidence   = extractConfidence(result, (int) classIdx);
            String rootCause    = classIdx < rootCauseLabels.size()
                                  ? rootCauseLabels.get((int) classIdx)
                                  : "UNKNOWN_" + classIdx;

            input.close();
            result.close();

            return RootCausePrediction.builder()
                .rootCause(rootCause)
                .confidence(confidence)
                .neId(context.getNeId())
                .alarmCode(context.getAlarmCode())
                .modelVersion(currentModelVersion)
                .build();

        } catch (OrtException e) {
            log.error("Root cause inference failed: {}", e.getMessage());
            return RootCausePrediction.unknown("Inference error: " + e.getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────
    // SEQUENCE MODEL (LSTM) — root cause from alarm sequence
    // ─────────────────────────────────────────────────────────────
    private static final int PAD_IDX = 0;
    private static final int UNK_IDX = 1;

    /**
     * Predict root cause from a sequence of alarm codes (e.g. last 20 alarms in time order).
     * Use when you have an incident or recent alarm burst; complements XGBoost per-NE classifier.
     */
    public RootCausePrediction predictRootCauseFromSequence(List<String> alarmCodes) {
        OrtSession session = sequenceSession.get();
        if (session == null || sequenceVocab == null || sequenceLabels == null || sequenceLabels.isEmpty()) {
            return RootCausePrediction.unknown("Sequence model not loaded");
        }
        try {
            long[] indices = buildSequenceIndices(alarmCodes);
            long[] shape = {1L, MAX_SEQ_LEN};
            OnnxTensor input = OnnxTensor.createTensor(env, indices, shape);
            Map<String, OnnxTensor> inputMap = Map.of("alarm_sequence", input);
            OrtSession.Result result = session.run(inputMap);
            long[] logitsShape = (long[]) result.get(0).getInfo().getShape();
            Object logitsObj = result.get(0).getValue();
            input.close();
            result.close();

            float[] logits = flattenFloatArray(logitsObj);
            if (logits == null || logits.length == 0) return RootCausePrediction.unknown("Invalid sequence output");
            int bestIdx = 0;
            float maxVal = logits[0];
            for (int i = 1; i < logits.length && i < sequenceLabels.size(); i++) {
                if (logits[i] > maxVal) { maxVal = logits[i]; bestIdx = i; }
            }
            float confidence = (float) (1.0 / (1.0 + Math.exp(-maxVal)));  // approximate from logit
            if (confidence < 0.1f) confidence = 0.5f;
            String rootCause = bestIdx < sequenceLabels.size() ? sequenceLabels.get(bestIdx) : "UNKNOWN";
            return RootCausePrediction.builder()
                .rootCause(rootCause)
                .confidence(confidence)
                .modelVersion(currentModelVersion)
                .build();
        } catch (OrtException e) {
            log.error("Sequence inference failed: {}", e.getMessage());
            return RootCausePrediction.unknown("Sequence error: " + e.getMessage());
        }
    }

    private long[] buildSequenceIndices(List<String> alarmCodes) {
        long[] out = new long[MAX_SEQ_LEN];
        int n = alarmCodes != null ? alarmCodes.size() : 0;
        int take = Math.min(n, MAX_SEQ_LEN);
        int pad = MAX_SEQ_LEN - take;
        int start = n - take;  // last 'take' alarms (most recent window)
        for (int i = 0; i < pad; i++) out[i] = PAD_IDX;
        for (int i = 0; i < take; i++) {
            String code = String.valueOf(alarmCodes.get(start + i)).trim();
            out[pad + i] = sequenceVocab.getOrDefault(code, UNK_IDX);
        }
        return out;
    }

    private static float[] flattenFloatArray(Object arr) {
        if (arr instanceof float[]) return (float[]) arr;
        if (arr instanceof Object[] o) {
            for (Object x : o) {
                if (x instanceof float[] f) return f;
            }
        }
        return null;
    }

    // ─────────────────────────────────────────────────────────────
    // ALARM PROPAGATION PREDICTION
    // ─────────────────────────────────────────────────────────────
    @SuppressWarnings("unchecked")
    public List<PropagationPrediction> predictPropagation(String alarmCode) {
        Map<String, Object> rules = propagationRules.get();
        if (rules == null || !rules.containsKey(alarmCode)) {
            return Collections.emptyList();
        }

        List<Map<String, Object>> consequents =
            (List<Map<String, Object>>) rules.get(alarmCode);

        List<PropagationPrediction> predictions = new ArrayList<>();

        for (Map<String, Object> rule : consequents) {
            double confidence = ((Number) rule.get("confidence")).doubleValue();
            if (confidence < minPropagationConfidence) continue;

            Number avgDelayNum = (Number) rule.get("avg_delay_sec");

            predictions.add(PropagationPrediction.builder()
                .alarmCode((String) rule.get("consequent"))
                .confidence(confidence)
                .avgDelaySec(avgDelayNum != null ? avgDelayNum.doubleValue() : null)
                .build());
        }

        return predictions;
    }

    // ─────────────────────────────────────────────────────────────
    // KPI ANOMALY DETECTION
    // ─────────────────────────────────────────────────────────────
    public AnomalyResult detectKpiAnomaly(KpiWindow window) {
        OrtSession session = anomalySession.get();
        if (session == null) {
            return AnomalyResult.normal();
        }

        try {
            float[] features = buildAnomalyFeatures(window);
            long[] shape = {1L, (long) features.length};
            OnnxTensor input = OnnxTensor.createTensor(env, features, shape);

            Map<String, OnnxTensor> inputMap = Map.of(
                session.getInputNames().iterator().next(), input
            );

            OrtSession.Result result = session.run(inputMap);

            // IsolationForest output: -1 = anomaly, 1 = normal
            long   prediction = ((long[]) result.get(0).getValue())[0];
            boolean isAnomaly = prediction == -1L;

            input.close();
            result.close();

            return AnomalyResult.builder()
                .isAnomaly(isAnomaly)
                .neId(window.getNeId())
                .kpiName(window.getKpiName())
                .currentValue(window.getCurrentValue())
                .build();

        } catch (OrtException e) {
            log.error("Anomaly inference failed: {}", e.getMessage());
            return AnomalyResult.normal();
        }
    }

    // ─────────────────────────────────────────────────────────────
    // FEATURE BUILDERS — map alarm context to float array
    // ─────────────────────────────────────────────────────────────
    private float[] buildRootCauseFeatures(AlarmContext ctx) {
        float[] f = new float[rootCauseFeatures.size()];
        Map<String, Float> values = ctx.toFeatureMap();
        for (int i = 0; i < rootCauseFeatures.size(); i++) {
            f[i] = values.getOrDefault(rootCauseFeatures.get(i), 0f);
        }
        return f;
    }

    private float[] buildAnomalyFeatures(KpiWindow window) {
        float[] f = new float[anomalyFeatures.size()];
        Map<String, Float> values = window.toFeatureMap();
        for (int i = 0; i < anomalyFeatures.size(); i++) {
            f[i] = values.getOrDefault(anomalyFeatures.get(i), 0f);
        }
        return f;
    }

    private float extractConfidence(OrtSession.Result result, int classIdx)
            throws OrtException {
        try {
            // Probabilities are in result[1] as Map<Long, Float>
            Object probObj = result.get(1).getValue();
            if (probObj instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<Long, Float> probs = (Map<Long, Float>) probObj;
                return probs.getOrDefault((long) classIdx, 0f);
            }
        } catch (Exception e) {
            log.debug("Could not extract confidence: {}", e.getMessage());
        }
        return 0.5f;  // default confidence if extraction fails
    }

    private void closeSession(OrtSession session) {
        if (session != null) {
            try { session.close(); } catch (OrtException ignored) {}
        }
    }

    private void startModelWatcher() {
        log.info("Model watcher started — checking every {}ms for updates", watcherIntervalMs);
    }
}


// ═══════════════════════════════════════════════════════════
// AlarmContext.java — input to root cause classifier
// ═══════════════════════════════════════════════════════════
/*
package com.noc.ml.model;

import lombok.*;
import java.util.*;

@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class AlarmContext {
    private String  alarmCode;
    private Long    neId;

    // From NETWORK_ELEMENT join
    private Integer neTypeEnc;
    private Integer technologyEnc;
    private Integer vendorEnc;
    private Integer domainEnc;
    private Integer neStatusEnc;
    private Integer operationalStateEnc;
    private Integer categoryEnc;

    // NE age / warranty
    private Double  neAgeDays;
    private Double  warrantyRemainingDays;
    private Integer warrantyExpired;
    private Integer techGeneration;
    private Integer isVirtual;

    // Protocol health
    private Integer bgpStatusNum;
    private Integer ospfStatusNum;
    private Integer lldpStatusNum;
    private Integer protocolHealthScore;

    // Graph topology
    private Double  graphDegree;
    private Double  betweenness;
    private Double  clusteringCoef;
    private Integer componentSize;
    private Integer isLeaf;
    private Integer topoLayer;

    // Hierarchy
    private Integer hierarchyDepth;
    private Integer isRootNode;
    private Integer geoCompleteness;
    private Integer geoL1;
    private Integer geoL2;
    private Integer geoL3;
    private Integer geoL4;

    // ISIS link health
    private Double  srcMaxUtilization;
    private Double  srcMaxErrorRate;
    private Double  srcMaxDropRate;
    private Integer srcCriticalLinks;
    private Double  linkHealthScore;

    // RAN
    private Integer azimuth;
    private Integer electricalTilt;
    private Integer mechanicalTilt;

    public Map<String, Float> toFeatureMap() {
        Map<String, Float> m = new HashMap<>();
        m.put("NE_TYPE_ENC",            safeFloat(neTypeEnc));
        m.put("TECHNOLOGY_ENC",         safeFloat(technologyEnc));
        m.put("VENDOR_ENC",             safeFloat(vendorEnc));
        m.put("DOMAIN_ENC",             safeFloat(domainEnc));
        m.put("NE_STATUS_ENC",          safeFloat(neStatusEnc));
        m.put("OPERATIONAL_STATE_ENC",  safeFloat(operationalStateEnc));
        m.put("CATEGORY_ENC",           safeFloat(categoryEnc));
        m.put("NE_AGE_DAYS",            safeFloat(neAgeDays));
        m.put("WARRANTY_REMAINING_DAYS",safeFloat(warrantyRemainingDays));
        m.put("WARRANTY_EXPIRED",       safeFloat(warrantyExpired));
        m.put("TECH_GENERATION",        safeFloat(techGeneration));
        m.put("IS_VIRTUAL_NUM",         safeFloat(isVirtual));
        m.put("BGP_STATUS_NUM",         safeFloat(bgpStatusNum));
        m.put("OSPF_STATUS_NUM",        safeFloat(ospfStatusNum));
        m.put("LLDP_STATUS_NUM",        safeFloat(lldpStatusNum));
        m.put("PROTOCOL_HEALTH_SCORE",  safeFloat(protocolHealthScore));
        m.put("GRAPH_DEGREE",           safeFloat(graphDegree));
        m.put("BETWEENNESS",            safeFloat(betweenness));
        m.put("CLUSTERING_COEF",        safeFloat(clusteringCoef));
        m.put("COMPONENT_SIZE",         safeFloat(componentSize));
        m.put("IS_LEAF",                safeFloat(isLeaf));
        m.put("TOPO_LAYER",             safeFloat(topoLayer));
        m.put("HIERARCHY_DEPTH",        safeFloat(hierarchyDepth));
        m.put("IS_ROOT_NODE",           safeFloat(isRootNode));
        m.put("GEO_COMPLETENESS",       safeFloat(geoCompleteness));
        m.put("GEOGRAPHY_L1_ID_FK",     safeFloat(geoL1));
        m.put("GEOGRAPHY_L2_ID_FK",     safeFloat(geoL2));
        m.put("GEOGRAPHY_L3_ID_FK",     safeFloat(geoL3));
        m.put("GEOGRAPHY_L4_ID_FK",     safeFloat(geoL4));
        m.put("SRC_MAX_UTILIZATION",    safeFloat(srcMaxUtilization));
        m.put("SRC_MAX_ERROR_RATE",     safeFloat(srcMaxErrorRate));
        m.put("SRC_MAX_DROP_RATE",      safeFloat(srcMaxDropRate));
        m.put("SRC_CRITICAL_LINKS",     safeFloat(srcCriticalLinks));
        m.put("LINK_HEALTH_SCORE",      safeFloat(linkHealthScore));
        m.put("AZIMUTH",                safeFloat(azimuth));
        m.put("ELECTRICAL_TILT",        safeFloat(electricalTilt));
        m.put("MECHANICAL_TILT",        safeFloat(mechanicalTilt));
        return m;
    }

    private float safeFloat(Number n) {
        return n != null ? n.floatValue() : 0f;
    }
}
*/
