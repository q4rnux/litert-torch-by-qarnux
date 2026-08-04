import React, { useState, useEffect, useRef } from "react";
import { ScrollView, Text, View, TouchableOpacity, Platform } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { ScreenContainer } from "@/components/screen-container";
import { useColors } from "@/hooks/use-colors";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withRepeat,
  Easing,
  withSequence,
} from "react-native-reanimated";
import { StyleSheet } from "react-native";

interface AgentNode {
  id: string;
  name: string;
  agent: string;
  description: string;
  icon: string;
  parallel?: boolean;
}

const PIPELINE_STAGES: AgentNode[] = [
  {
    id: "parser",
    name: "Parser",
    agent: "ParserAgent",
    description: "Parses the input GGUF model file, extracting metadata, tensor information, and quantization parameters.",
    icon: "document-text-outline",
  },
  {
    id: "dequant",
    name: "Dequantization",
    agent: "DequantizationAgent",
    description: "Handles dequantization of model weights based on the quantization configuration, preparing tensors for conversion.",
    icon: "swap-horizontal-outline",
  },
  {
    id: "authoring",
    name: "Model Authoring",
    agent: "ModelAuthoringAgent",
    description: "Constructs the LiteRT-LM model architecture using Google's LiteRT runtime APIs.",
    icon: "construct-outline",
  },
  {
    id: "tokenizer",
    name: "Tokenizer",
    agent: "TokenizerAgent",
    description: "Converts and embeds the model's tokenizer into the LiteRT-LM format. Runs in parallel with ModelAuthoringAgent.",
    icon: "text-outline",
    parallel: true,
  },
  {
    id: "metadata",
    name: "Metadata",
    agent: "MetadataAgent",
    description: "Assembles model metadata including behavior profiles, template configs, skill.md, and system prompts.",
    icon: "information-circle-outline",
  },
  {
    id: "conversion",
    name: "Conversion",
    agent: "ConversionAgent",
    description: "Performs the final conversion of all components into the .litertlm format, including quantization.",
    icon: "sync-outline",
  },
  {
    id: "packaging",
    name: "Packaging",
    agent: "PackagingAgent",
    description: "Packages the final .litertlm file with all assets, metadata, and configuration for distribution.",
    icon: "cube-outline",
  },
];

function AnimatedNode({
  node,
  index,
  isComplete,
  isRunning,
  onPress,
}: {
  node: AgentNode;
  index: number;
  isComplete: boolean;
  isRunning: boolean;
  onPress: () => void;
}) {
  const colors = useColors();
  const pulseOpacity = useSharedValue(0);
  const scaleVal = useSharedValue(1);

  useEffect(() => {
    if (isRunning) {
      pulseOpacity.value = withRepeat(
        withSequence(
          withTiming(1, { duration: 800, easing: Easing.inOut(Easing.ease) }),
          withTiming(0, { duration: 800, easing: Easing.inOut(Easing.ease) })
        ),
        -1,
        false
      );
      scaleVal.value = withRepeat(
        withSequence(
          withTiming(1.05, { duration: 800 }),
          withTiming(1, { duration: 800 })
        ),
        -1,
        false
      );
    } else if (isComplete) {
      pulseOpacity.value = withTiming(0, { duration: 300 });
      scaleVal.value = withTiming(1, { duration: 300 });
    }
  }, [isRunning, isComplete]);

  const pulseStyle = useAnimatedStyle(() => ({
    opacity: pulseOpacity.value,
  }));

  const scaleStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scaleVal.value }],
  }));

  const bgColor = isComplete
    ? colors.success
    : isRunning
    ? colors.primary
    : colors.surface;

  const borderColor = isComplete
    ? colors.success
    : isRunning
    ? colors.primary
    : colors.border;

  const textColor = isComplete || isRunning ? "#fff" : colors.foreground;

  return (
    <Animated.View style={[styles.nodeContainer, scaleStyle]}>
      {node.parallel && (
        <View style={[styles.parallelBadge, { backgroundColor: colors.warning }]}>
          <Text style={styles.parallelBadgeText}>PARALLEL</Text>
        </View>
      )}

      {/* Pulse ring for running state */}
      {isRunning && (
        <Animated.View
          style={[
            styles.pulseRing,
            pulseStyle,
            { borderColor: colors.primary, backgroundColor: colors.primary + "20" },
          ]}
        />
      )}

      <TouchableOpacity
        onPress={onPress}
        style={[styles.node, { backgroundColor: bgColor, borderColor }]}
        activeOpacity={0.7}
      >
        <Ionicons name={node.icon as any} size={24} color={textColor} />
        <Text style={[styles.nodeLabel, { color: textColor }]}>{node.name}</Text>
        <Text style={[styles.nodeAgent, { color: textColor + "AA" }]}>{node.agent}</Text>

        {/* Status indicator */}
        <View style={[styles.statusDot, { backgroundColor: isComplete ? "#fff" : isRunning ? "#fff" : colors.muted }]} />
      </TouchableOpacity>

      {/* Arrow connector */}
      {index < PIPELINE_STAGES.length - 1 && !PIPELINE_STAGES[index + 1]?.parallel && (
        <View style={styles.arrow}>
          <Ionicons name="arrow-down" size={20} color={isComplete ? colors.success : colors.muted} />
        </View>
      )}
    </Animated.View>
  );
}

export default function PipelineScreen() {
  const colors = useColors();
  const [activeStage, setActiveStage] = useState<string | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [completedStages, setCompletedStages] = useState<Set<string>>(new Set());
  const [runningStage, setRunningStage] = useState<string | null>(null);

  // Simulation effect
  useEffect(() => {
    if (!simulating) return;
    let currentIdx = 0;
    const interval = setInterval(() => {
      if (currentIdx >= PIPELINE_STAGES.length) {
        setSimulating(false);
        setRunningStage(null);
        clearInterval(interval);
        return;
      }
      setRunningStage(PIPELINE_STAGES[currentIdx].id);
      setTimeout(() => {
        setCompletedStages((prev) => {
          const next = new Set(prev);
          next.add(PIPELINE_STAGES[currentIdx].id);
          return next;
        });
      }, 1200);
      currentIdx++;
    }, 2000);
    return () => clearInterval(interval);
  }, [simulating]);

  const handleSimulate = () => {
    setCompletedStages(new Set());
    setRunningStage(null);
    setActiveStage(null);
    setSimulating(true);
  };

  const handleReset = () => {
    setSimulating(false);
    setCompletedStages(new Set());
    setRunningStage(null);
    setActiveStage(null);
  };

  return (
    <ScreenContainer className="flex-1">
      <ScrollView contentContainerStyle={{ padding: 16 }} style={{ flex: 1 }}>
        {/* Header */}
        <View style={styles.headerRow}>
          <Text style={[styles.headerTitle, { color: colors.foreground }]}>Pipeline</Text>
          <Text style={[styles.headerSubtitle, { color: colors.muted }]}>
            7-Stage Orchestration
          </Text>
        </View>

        {/* Control Buttons */}
        <View style={styles.controlsRow}>
          <TouchableOpacity
            onPress={handleSimulate}
            disabled={simulating}
            style={[
              styles.controlBtn,
              {
                backgroundColor: simulating ? colors.muted : colors.primary,
                opacity: simulating ? 0.5 : 1,
              },
            ]}
          >
            <Ionicons name="play-circle-outline" size={18} color="#fff" />
            <Text style={styles.controlBtnText}>Simulate</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={handleReset}
            style={[styles.controlBtn, { backgroundColor: colors.border }]}
          >
            <Ionicons name="refresh-circle-outline" size={18} color={colors.foreground} />
            <Text style={[styles.controlBtnText, { color: colors.foreground }]}>Reset</Text>
          </TouchableOpacity>
        </View>

        {/* Pipeline Flow */}
        <View style={styles.pipelineContainer}>
          {PIPELINE_STAGES.map((stage, index) => (
            <AnimatedNode
              key={stage.id}
              node={stage}
              index={index}
              isComplete={completedStages.has(stage.id)}
              isRunning={runningStage === stage.id}
              onPress={() => setActiveStage(activeStage === stage.id ? null : stage.id)}
            />
          ))}
        </View>

        {/* Active Stage Detail */}
        {activeStage && (
          <View style={[styles.detailCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
            <Text style={[styles.detailTitle, { color: colors.foreground }]}>
              {PIPELINE_STAGES.find((s) => s.id === activeStage)?.name}
            </Text>
            <Text style={[styles.detailAgent, { color: colors.primary }]}>
              {PIPELINE_STAGES.find((s) => s.id === activeStage)?.agent}
            </Text>
            <Text style={[styles.detailDesc, { color: colors.muted }]}>
              {PIPELINE_STAGES.find((s) => s.id === activeStage)?.description}
            </Text>

            {/* Status */}
            <View style={styles.detailStatus}>
              <View
                style={[
                  styles.statusPill,
                  {
                    backgroundColor: completedStages.has(activeStage)
                      ? colors.success + "20"
                      : runningStage === activeStage
                      ? colors.primary + "20"
                      : colors.muted + "20",
                  },
                ]}
              >
                <Text
                  style={[
                    styles.statusPillText,
                    {
                      color: completedStages.has(activeStage)
                        ? colors.success
                        : runningStage === activeStage
                        ? colors.primary
                        : colors.muted,
                    },
                  ]}
                >
                  {completedStages.has(activeStage) ? "Complete" : runningStage === activeStage ? "Running" : "Pending"}
                </Text>
              </View>
            </View>
          </View>
        )}

        {/* Info Footer */}
        <View style={[styles.infoCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.infoTitle, { color: colors.foreground }]}>About the Pipeline</Text>
          <Text style={[styles.infoText, { color: colors.muted }]}>
            The litert-torch-by-qarnux orchestrator coordinates 7 specialized agents to convert
            GGUF model files into Google's LiteRT-LM format (.litertlm). Each agent handles a
            specific stage of the conversion process, with ModelAuthoringAgent and TokenizerAgent
            running in parallel for efficiency.
          </Text>
        </View>

        <View style={{ height: 24 }} />
      </ScrollView>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  headerRow: { marginBottom: 16, marginTop: 8 },
  headerTitle: { fontSize: 24, fontWeight: "700" },
  headerSubtitle: { fontSize: 13, marginTop: 2 },
  controlsRow: { flexDirection: "row", gap: 12, marginBottom: 20 },
  controlBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  controlBtnText: { color: "#fff", fontSize: 14, fontWeight: "600" },
  pipelineContainer: { alignItems: "center", paddingVertical: 8 },
  nodeContainer: { alignItems: "center", position: "relative" },
  parallelBadge: {
    position: "absolute",
    top: -8,
    right: 8,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
    zIndex: 2,
  },
  parallelBadgeText: { color: "#fff", fontSize: 9, fontWeight: "700", letterSpacing: 0.5 },
  pulseRing: {
    position: "absolute",
    width: 160,
    height: 90,
    borderRadius: 45,
    borderWidth: 2,
  },
  node: {
    width: 150,
    paddingVertical: 14,
    paddingHorizontal: 12,
    borderRadius: 14,
    borderWidth: 2,
    alignItems: "center",
    gap: 4,
  },
  nodeLabel: { fontSize: 13, fontWeight: "600", textAlign: "center" },
  nodeAgent: { fontSize: 10, fontWeight: "500" },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 2,
  },
  arrow: { paddingVertical: 6 },
  detailCard: {
    borderWidth: 1,
    borderRadius: 14,
    padding: 16,
    marginTop: 16,
  },
  detailTitle: { fontSize: 16, fontWeight: "700", marginBottom: 4 },
  detailAgent: { fontSize: 13, fontWeight: "600", marginBottom: 8 },
  detailDesc: { fontSize: 13, lineHeight: 20 },
  detailStatus: { marginTop: 12, alignItems: "center" },
  statusPill: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 12 },
  statusPillText: { fontSize: 13, fontWeight: "600" },
  infoCard: {
    borderWidth: 1,
    borderRadius: 14,
    padding: 16,
    marginTop: 16,
  },
  infoTitle: { fontSize: 14, fontWeight: "600", marginBottom: 8 },
  infoText: { fontSize: 13, lineHeight: 20 },
});
