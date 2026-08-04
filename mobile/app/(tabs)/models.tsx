import React, { useState } from "react";
import {
  ScrollView,
  Text,
  View,
  TouchableOpacity,
  Modal,
  TextInput,
  Switch,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { ScreenContainer } from "@/components/screen-container";
import { useAppState } from "@/hooks/use-app-state";
import { useColors } from "@/hooks/use-colors";
import {
  SUPPORTED_ARCHITECTURES,
  QUANTIZATION_RECIPES,
  ModelArchitecture,
  ChatTemplate,
} from "@/lib/types";
import { StyleSheet } from "react-native";

function TemplateBadge({ name, active, onPress, color }: { name: string; active: boolean; onPress: () => void; color: any }) {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={[
        styles.templateBadge,
        {
          borderColor: active ? color.primary : color.border,
          backgroundColor: active ? color.primary + "20" : color.surface,
        },
      ]}
    >
      <Text style={[styles.templateBadgeText, { color: active ? color.primary : color.foreground }]}>
        {name}
      </Text>
    </TouchableOpacity>
  );
}

export default function ModelsScreen() {
  const colors = useColors();
  const appState = useAppState();
  const [selectedArch, setSelectedArch] = useState<ModelArchitecture | null>(null);
  const [showQuantConfig, setShowQuantConfig] = useState(false);

  const handleSelectArch = (arch: ModelArchitecture) => {
    setSelectedArch(selectedArch?.id === arch.id ? null : arch);
  };

  return (
    <ScreenContainer className="flex-1">
      <ScrollView contentContainerStyle={{ padding: 16 }} style={{ flex: 1 }}>
        {/* Header */}
        <View style={styles.headerRow}>
          <Text style={[styles.headerTitle, { color: colors.foreground }]}>Models</Text>
          <Text style={[styles.headerSubtitle, { color: colors.muted }]}>
            Architecture Library
          </Text>
        </View>

        {/* Architecture Cards */}
        {SUPPORTED_ARCHITECTURES.map((arch) => {
          const isSelected = selectedArch?.id === arch.id;
          return (
            <TouchableOpacity
              key={arch.id}
              onPress={() => handleSelectArch(arch)}
              style={[
                styles.archCard,
                {
                  borderColor: isSelected ? colors.primary : colors.border,
                  backgroundColor: colors.surface,
                  borderWidth: isSelected ? 2 : 1,
                },
              ]}
            >
              <View style={styles.archHeader}>
                <Ionicons
                  name="cube-outline"
                  size={24}
                  color={isSelected ? colors.primary : colors.muted}
                />
                <View style={{ flex: 1, marginLeft: 12 }}>
                  <Text style={[styles.archName, { color: colors.foreground }]}>{arch.name}</Text>
                  <Text style={[styles.archDesc, { color: colors.muted }]}>{arch.description}</Text>
                </View>
                <Ionicons
                  name={isSelected ? "chevron-up" : "chevron-down"}
                  size={20}
                  color={colors.muted}
                />
              </View>

              {isSelected && (
                <View style={styles.archDetail}>
                  <Text style={[styles.detailLabel, { color: colors.muted }]}>SUPPORTED TEMPLATES</Text>
                  <View style={styles.templateRow}>
                    {arch.templates.map((tmpl) => (
                      <TemplateBadge
                        key={tmpl}
                        name={tmpl}
                        active={appState.template.chat_template === tmpl}
                        onPress={() => appState.updateTemplate({ chat_template: tmpl })}
                        color={colors}
                      />
                    ))}
                  </View>
                </View>
              )}
            </TouchableOpacity>
          );
        })}

        {/* Quantization Config */}
        <TouchableOpacity
          onPress={() => setShowQuantConfig(true)}
          style={[styles.quantCard, { borderColor: colors.border, backgroundColor: colors.surface }]}
        >
          <View style={styles.quantHeader}>
            <Ionicons name="layers-outline" size={22} color={colors.primary} />
            <Text style={[styles.quantTitle, { color: colors.foreground }]}>Quantization Config</Text>
            <Ionicons name="chevron-forward" size={16} color={colors.muted} />
          </View>
          <View style={styles.quantSummary}>
            <Text style={[styles.quantSummaryText, { color: colors.muted }]}>
              {appState.quantization.recipe} • {appState.quantization.output_dtype} • gs:{appState.quantization.group_size}
            </Text>
          </View>
        </TouchableOpacity>

        <View style={{ height: 24 }} />
      </ScrollView>

      {/* Quantization Modal */}
      <Modal visible={showQuantConfig} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.foreground }]}>Quantization Configuration</Text>
              <TouchableOpacity onPress={() => setShowQuantConfig(false)}>
                <Ionicons name="close" size={24} color={colors.muted} />
              </TouchableOpacity>
            </View>

            <ScrollView style={{ flex: 1, paddingHorizontal: 16, paddingBottom: 24 }}>
              {/* Recipe Selector */}
              <Text style={[styles.fieldLabel, { color: colors.muted }]}>RECIPE</Text>
              <View style={styles.recipeGrid}>
                {QUANTIZATION_RECIPES.map((recipe) => (
                  <TouchableOpacity
                    key={recipe}
                    onPress={() => appState.updateQuantization({ recipe })}
                    style={[
                      styles.recipeBtn,
                      {
                        borderColor: appState.quantization.recipe === recipe ? colors.primary : colors.border,
                        backgroundColor: appState.quantization.recipe === recipe ? colors.primary + "15" : colors.surface,
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.recipeText,
                        { color: appState.quantization.recipe === recipe ? colors.primary : colors.foreground },
                      ]}
                    >
                      {recipe}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              {/* Method */}
              <Text style={[styles.fieldLabel, { color: colors.muted }]}>METHOD</Text>
              <View style={styles.inputRow}>
                <TextInput
                  value={appState.quantization.method}
                  onChangeText={(text) => appState.updateQuantization({ method: text })}
                  style={[styles.input, { color: colors.foreground, borderColor: colors.border, backgroundColor: colors.background }]}
                />
              </View>

              {/* Output Dtype */}
              <Text style={[styles.fieldLabel, { color: colors.muted }]}>OUTPUT DTYPE</Text>
              <View style={styles.dtypeRow}>
                {["int4", "int8", "fp16", "fp32", "fp8"].map((dtype) => (
                  <TouchableOpacity
                    key={dtype}
                    onPress={() => appState.updateQuantization({ output_dtype: dtype })}
                    style={[
                      styles.dtypeBtn,
                      {
                        borderColor: appState.quantization.output_dtype === dtype ? colors.primary : colors.border,
                        backgroundColor: appState.quantization.output_dtype === dtype ? colors.primary + "15" : colors.surface,
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.dtypeText,
                        { color: appState.quantization.output_dtype === dtype ? colors.primary : colors.foreground },
                      ]}
                    >
                      {dtype}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              {/* Group Size */}
              <Text style={[styles.fieldLabel, { color: colors.muted }]}>GROUP SIZE</Text>
              <TextInput
                value={appState.quantization.group_size.toString()}
                onChangeText={(text) => {
                  const num = parseInt(text);
                  if (!isNaN(num) && num > 0) {
                    appState.updateQuantization({ group_size: num });
                  }
                }}
                keyboardType="number-pad"
                style={[styles.input, { color: colors.foreground, borderColor: colors.border, backgroundColor: colors.background }]}
              />

              {/* Per Channel */}
              <View style={styles.switchRow}>
                <Text style={[styles.switchLabel, { color: colors.foreground }]}>Per Channel</Text>
                <Switch
                  value={appState.quantization.per_channel}
                  onValueChange={(val) => appState.updateQuantization({ per_channel: val })}
                  trackColor={{ false: colors.border, true: colors.primary }}
                  thumbColor={colors.primary}
                />
              </View>

              {/* Symmetric */}
              <View style={styles.switchRow}>
                <Text style={[styles.switchLabel, { color: colors.foreground }]}>Symmetric</Text>
                <Switch
                  value={appState.quantization.symmetric}
                  onValueChange={(val) => appState.updateQuantization({ symmetric: val })}
                  trackColor={{ false: colors.border, true: colors.primary }}
                  thumbColor={colors.primary}
                />
              </View>
            </ScrollView>
          </View>
        </View>
      </Modal>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  headerRow: { marginBottom: 16, marginTop: 8 },
  headerTitle: { fontSize: 24, fontWeight: "700" },
  headerSubtitle: { fontSize: 13, marginTop: 2 },
  archCard: {
    borderWidth: 1,
    borderRadius: 14,
    padding: 14,
    marginBottom: 10,
  },
  archHeader: { flexDirection: "row", alignItems: "center" },
  archName: { fontSize: 16, fontWeight: "700" },
  archDesc: { fontSize: 12, marginTop: 2 },
  archDetail: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.05)" },
  detailLabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 1.2,
    marginBottom: 8,
  },
  templateRow: { flexDirection: "row", gap: 8, flexWrap: "wrap" },
  templateBadge: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  templateBadgeText: { fontSize: 12, fontWeight: "600" },
  quantCard: {
    borderWidth: 1,
    borderRadius: 14,
    padding: 14,
    marginTop: 8,
  },
  quantHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  quantTitle: { flex: 1, fontSize: 15, fontWeight: "600" },
  quantSummary: { marginTop: 8 },
  quantSummaryText: { fontSize: 12 },
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "flex-end" },
  modalContent: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderWidth: 1,
    maxHeight: "90%",
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.1)",
  },
  modalTitle: { fontSize: 18, fontWeight: "700" },
  fieldLabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 1.2,
    marginTop: 14,
    marginBottom: 8,
  },
  recipeGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  recipeBtn: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  recipeText: { fontSize: 12, fontWeight: "600" },
  inputRow: { marginBottom: 4 },
  input: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
  },
  dtypeRow: { flexDirection: "row", gap: 8, flexWrap: "wrap" },
  dtypeBtn: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  dtypeText: { fontSize: 13, fontWeight: "600" },
  switchRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    marginTop: 4,
  },
  switchLabel: { fontSize: 14, fontWeight: "500" },
});
