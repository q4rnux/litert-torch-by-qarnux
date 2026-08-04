import React, { useState } from "react";
import {
  ScrollView,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  Modal,
  Alert,
  Share,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { ScreenContainer } from "@/components/screen-container";
import { useAppState } from "@/hooks/use-app-state";
import { useColors } from "@/hooks/use-colors";
import { useColorScheme } from "@/hooks/use-color-scheme";
import { StyleSheet } from "react-native";

export default function SettingsScreen() {
  const colors = useColors();
  const colorScheme = useColorScheme();
  const appState = useAppState();
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportName, setExportName] = useState("");
  const [exportFormat, setExportFormat] = useState<"json" | "yaml">("json");

  const handleExport = () => {
    if (!exportName.trim()) {
      Alert.alert("Error", "Please enter a config name");
      return;
    }
    const data = appState.exportConfig(exportName.trim(), exportFormat);

    // Try to share/copy the config
    if (Platform.OS !== "web") {
      Share.share({
        message: data,
        title: exportName,
      }).catch(() => {
        // Share cancelled - config is already saved
      });
    } else {
      Alert.alert("Exported", `Configuration "${exportName}" saved in ${exportFormat.toUpperCase()} format. Check the Saved Configs section below.`);
    }

    setExportName("");
    setShowExportModal(false);
  };

  const handleCopyConfig = (config: { name: string; data: string; format: string }) => {
    if (Platform.OS !== "web") {
      try {
        const Clipboard = require("expo-clipboard");
        Clipboard.setStringAsync(config.data);
        Alert.alert("Copied", "Configuration copied to clipboard");
      } catch {
        Alert.alert("Exported", "Configuration saved. Check the Saved Configs section.");
      }
    } else {
      if (typeof navigator !== "undefined" && navigator.clipboard) {
        navigator.clipboard.writeText(config.data);
      }
      Alert.alert("Copied", "Configuration copied to clipboard");
    }
  };

  return (
    <ScreenContainer className="flex-1">
      <ScrollView contentContainerStyle={{ padding: 16 }} style={{ flex: 1 }}>
        {/* Header */}
        <View style={styles.headerRow}>
          <Text style={[styles.headerTitle, { color: colors.foreground }]}>Settings</Text>
          <Text style={[styles.headerSubtitle, { color: colors.muted }]}>
            Export & Configuration
          </Text>
        </View>

        {/* Export Configuration */}
        <View style={[styles.card, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.cardTitle, { color: colors.foreground }]}>Export Configuration</Text>
          <Text style={[styles.cardDesc, { color: colors.muted }]}>
            Export your behavior profile, template, and quantization settings as JSON or YAML for use with the litert-torch-by-qarnux CLI tool.
          </Text>
          <TouchableOpacity
            onPress={() => setShowExportModal(true)}
            style={[styles.exportBtn, { backgroundColor: colors.primary }]}
          >
            <Ionicons name="download-outline" size={20} color="#fff" />
            <Text style={styles.exportBtnText}>Export Configuration</Text>
          </TouchableOpacity>
        </View>

        {/* Saved Configs */}
        {appState.savedConfigs.length > 0 && (
          <View style={{ marginTop: 20 }}>
            <Text style={[styles.sectionTitle, { color: colors.muted }]}>SAVED CONFIGURATIONS</Text>
            {appState.savedConfigs.map((config, index) => (
              <View
                key={index}
                style={[styles.configCard, { borderColor: colors.border, backgroundColor: colors.surface }]}
              >
                <View style={styles.configHeader}>
                  <View style={{ flex: 1 }}>
                    <Text style={[styles.configName, { color: colors.foreground }]}>{config.name}</Text>
                    <Text style={[styles.configFormat, { color: colors.muted }]}>
                      {config.format.toUpperCase()} • {new Date().toLocaleDateString()}
                    </Text>
                  </View>
                  <View style={styles.configActions}>
                    <TouchableOpacity
                      onPress={() => handleCopyConfig(config)}
                      style={[styles.actionIcon, { backgroundColor: colors.primary + "15" }]}
                    >
                      <Ionicons name="copy-outline" size={16} color={colors.primary} />
                    </TouchableOpacity>
                    <TouchableOpacity
                      onPress={() => appState.deleteConfig(index)}
                      style={[styles.actionIcon, { backgroundColor: colors.error + "15" }]}
                    >
                      <Ionicons name="trash-outline" size={16} color={colors.error} />
                    </TouchableOpacity>
                  </View>
                </View>
                <View style={[styles.configPreview, { backgroundColor: colors.background }]}>
                  <Text
                    style={{
                      color: colors.muted,
                      fontSize: 11,
                      fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
                      lineHeight: 16,
                    }}
                    numberOfLines={3}
                  >
                    {config.data}
                  </Text>
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Quick Actions */}
        <Text style={[styles.sectionTitle, { color: colors.muted }]}>QUICK ACTIONS</Text>

        {/* Reset All */}
        <TouchableOpacity
          onPress={() => {
            Alert.alert("Reset", "Reset all configurations to defaults?", [
              { text: "Cancel", style: "cancel" },
              {
                text: "Reset",
                style: "destructive",
                onPress: () => {
                  // Trigger a re-render by changing active profile
                  const defaultProfile = appState.profiles.find((p) => p.name === "default");
                  if (defaultProfile) {
                    appState.setActiveProfile(defaultProfile.id);
                  }
                  appState.updateTemplate({
                    chat_template: "chatml",
                    system_prompt: "You are a helpful assistant.",
                    skill_md_content: "",
                    personality: "helpful",
                    role: "assistant",
                  });
                },
              },
            ]);
          }}
          style={[styles.actionRow, { borderColor: colors.border, backgroundColor: colors.surface }]}
        >
          <Ionicons name="refresh-outline" size={20} color={colors.warning} />
          <Text style={[styles.actionText, { color: colors.foreground }]}>Reset to Defaults</Text>
        </TouchableOpacity>

        {/* About */}
        <View style={[styles.aboutCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.aboutTitle, { color: colors.foreground }]}>TTS Prompter</Text>
          <Text style={[styles.aboutVersion, { color: colors.muted }]}>v1.0.0 • by qarnux</Text>
          <Text style={[styles.aboutDesc, { color: colors.muted }]}>
            Companion mobile app for litert-torch-by-qarnux. Configure model behavior profiles,
            templates, and quantization settings. Preview prompts with text-to-speech.
          </Text>
          <View style={styles.techTags}>
            <Text style={[styles.techTag, { backgroundColor: colors.primary + "20", color: colors.primary }]}>
              Expo
            </Text>
            <Text style={[styles.techTag, { backgroundColor: colors.primary + "20", color: colors.primary }]}>
              React Native
            </Text>
            <Text style={[styles.techTag, { backgroundColor: colors.primary + "20", color: colors.primary }]}>
              LiteRT-LM
            </Text>
          </View>
        </View>

        <View style={{ height: 24 }} />
      </ScrollView>

      {/* Export Modal */}
      <Modal visible={showExportModal} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.foreground }]}>Export Configuration</Text>
              <TouchableOpacity onPress={() => setShowExportModal(false)}>
                <Ionicons name="close" size={24} color={colors.muted} />
              </TouchableOpacity>
            </View>

            <View style={{ padding: 16 }}>
              <Text style={[styles.fieldLabel, { color: colors.muted }]}>CONFIG NAME</Text>
              <TextInput
                value={exportName}
                onChangeText={setExportName}
                placeholder="e.g., my_coding_profile"
                placeholderTextColor={colors.muted}
                style={[styles.input, { color: colors.foreground, borderColor: colors.border, backgroundColor: colors.background }]}
              />

              <Text style={[styles.fieldLabel, { color: colors.muted }]}>FORMAT</Text>
              <View style={styles.formatRow}>
                <TouchableOpacity
                  onPress={() => setExportFormat("json")}
                  style={[
                    styles.formatBtn,
                    {
                      borderColor: exportFormat === "json" ? colors.primary : colors.border,
                      backgroundColor: exportFormat === "json" ? colors.primary + "15" : colors.surface,
                    },
                  ]}
                >
                  <Text style={[styles.formatText, { color: exportFormat === "json" ? colors.primary : colors.foreground }]}>
                    JSON
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => setExportFormat("yaml")}
                  style={[
                    styles.formatBtn,
                    {
                      borderColor: exportFormat === "yaml" ? colors.primary : colors.border,
                      backgroundColor: exportFormat === "yaml" ? colors.primary + "15" : colors.surface,
                    },
                  ]}
                >
                  <Text style={[styles.formatText, { color: exportFormat === "yaml" ? colors.primary : colors.foreground }]}>
                    YAML
                  </Text>
                </TouchableOpacity>
              </View>

              <TouchableOpacity
                onPress={handleExport}
                style={[styles.exportConfirmBtn, { backgroundColor: colors.primary, marginTop: 20 }]}
              >
                <Text style={styles.exportConfirmText}>Export & Save</Text>
              </TouchableOpacity>
            </View>
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
  card: { borderWidth: 1, borderRadius: 14, padding: 16 },
  cardTitle: { fontSize: 16, fontWeight: "700", marginBottom: 6 },
  cardDesc: { fontSize: 13, lineHeight: 20, marginBottom: 14 },
  exportBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 12,
    borderRadius: 10,
  },
  exportBtnText: { color: "#fff", fontSize: 14, fontWeight: "600" },
  sectionTitle: {
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 1.2,
    marginTop: 20,
    marginBottom: 10,
  },
  configCard: {
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
  },
  configHeader: { flexDirection: "row", alignItems: "center" },
  configName: { fontSize: 14, fontWeight: "600" },
  configFormat: { fontSize: 11, marginTop: 2 },
  configActions: { flexDirection: "row", gap: 8 },
  actionIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  configPreview: {
    borderRadius: 8,
    padding: 10,
    marginTop: 10,
  },
  actionRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  actionText: { fontSize: 14, fontWeight: "500" },
  aboutCard: {
    borderWidth: 1,
    borderRadius: 14,
    padding: 16,
    marginTop: 20,
    alignItems: "center",
  },
  aboutTitle: { fontSize: 18, fontWeight: "700" },
  aboutVersion: { fontSize: 13, marginTop: 4 },
  aboutDesc: { fontSize: 13, lineHeight: 20, textAlign: "center", marginTop: 12 },
  techTags: { flexDirection: "row", gap: 8, marginTop: 14, flexWrap: "wrap", justifyContent: "center" },
  techTag: { paddingHorizontal: 12, paddingVertical: 4, borderRadius: 12, fontSize: 12, fontWeight: "600" },
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "flex-end" },
  modalContent: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderWidth: 1,
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
    marginBottom: 8,
    marginTop: 4,
  },
  input: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    marginBottom: 4,
  },
  formatRow: { flexDirection: "row", gap: 12 },
  formatBtn: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: "center",
  },
  formatText: { fontSize: 14, fontWeight: "600" },
  exportConfirmBtn: { paddingVertical: 14, borderRadius: 10, alignItems: "center" },
  exportConfirmText: { color: "#fff", fontSize: 15, fontWeight: "600" },
});
