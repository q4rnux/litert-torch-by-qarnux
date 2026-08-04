import React, { useState, useCallback, useMemo } from "react";
import {
  ScrollView,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  Modal,
  FlatList,
  Alert,
  Platform,
} from "react-native";
import * as Speech from "expo-speech";
import { Ionicons } from "@expo/vector-icons";
import { ScreenContainer } from "@/components/screen-container";
import { useAppState } from "@/hooks/use-app-state";
import { useColors } from "@/hooks/use-colors";
import { BEHAVIOR_CATEGORIES, BUILTIN_PRESETS, CHAT_TEMPLATES } from "@/lib/types";
import { StyleSheet } from "react-native";

export default function PrompterScreen() {
  const colors = useColors();
  const appState = useAppState();
  const [isPlaying, setIsPlaying] = useState(false);
  const [showProfilePicker, setShowProfilePicker] = useState(false);
  const [showTemplatePicker, setShowTemplatePicker] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

  const activeProfile = appState.profiles.find((p) => p.id === appState.activeProfileId) || appState.profiles[0];

  // Build the full system prompt preview
  const fullPrompt = useMemo(() => {
    const parts: string[] = [];
    if (appState.template.system_prompt) {
      parts.push(`[System Prompt]\n${appState.template.system_prompt}`);
    }
    if (appState.template.personality) {
      parts.push(`[Personality]\n${appState.template.personality}`);
    }
    if (appState.template.role) {
      parts.push(`[Role]\n${appState.template.role}`);
    }
    if (appState.template.skill_md_content) {
      parts.push(`[Skills]\n${appState.template.skill_md_content}`);
    }
    if (appState.promptText) {
      parts.push(`[User Prompt]\n${appState.promptText}`);
    }
    // Behavior emphasis summary
    const active = Object.entries(activeProfile.values)
      .filter(([_, v]) => v !== 0)
      .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v.toFixed(1)}`);
    if (active.length > 0) {
      parts.push(`[Behavior Profile: ${activeProfile.name}]\n${active.join(", ")}`);
    }
    return parts.join("\n\n");
  }, [appState.template, appState.promptText, activeProfile]);

  const handleTTS = useCallback(() => {
    if (isPlaying) {
      Speech.stop();
      setIsPlaying(false);
      return;
    }
    const textToRead = appState.promptText || fullPrompt || "No prompt configured.";
    setIsPlaying(true);
    Speech.speak(textToRead, {
      onDone: () => setIsPlaying(false),
      onError: () => setIsPlaying(false),
    });
  }, [isPlaying, appState.promptText, fullPrompt]);

  const handleToggleTTS = useCallback(() => {
    if (Platform.OS !== "web") {
      handleTTS();
    } else {
      Alert.alert("TTS", "Text-to-speech is available on iOS and Android devices.");
    }
  }, [handleTTS]);

  const applyPreset = useCallback((presetName: string) => {
    appState.applyProfilePreset(presetName);
    setShowProfilePicker(false);
  }, [appState]);

  return (
    <ScreenContainer className="flex-1">
      <ScrollView contentContainerStyle={{ padding: 16 }} style={{ flex: 1 }}>
        {/* Header */}
        <View style={styles.headerRow}>
          <Text style={[styles.headerTitle, { color: colors.foreground }]}>TTS Prompter</Text>
          <Text style={[styles.headerSubtitle, { color: colors.muted }]}>by qarnux</Text>
        </View>

        {/* Profile Selector */}
        <TouchableOpacity
          onPress={() => setShowProfilePicker(true)}
          style={[styles.selector, { borderColor: colors.border, backgroundColor: colors.surface }]}
        >
          <Ionicons name="settings-outline" size={20} color={colors.primary} />
          <Text style={[styles.selectorText, { color: colors.foreground }]}>
            Profile: {activeProfile.name}
          </Text>
          <Ionicons name="chevron-down" size={16} color={colors.muted} />
        </TouchableOpacity>

        {/* Template Selector */}
        <TouchableOpacity
          onPress={() => setShowTemplatePicker(true)}
          style={[styles.selector, { borderColor: colors.border, backgroundColor: colors.surface }]}
        >
          <Ionicons name="document-text-outline" size={20} color={colors.primary} />
          <Text style={[styles.selectorText, { color: colors.foreground }]}>
            Template: {appState.template.chat_template}
          </Text>
          <Ionicons name="chevron-down" size={16} color={colors.muted} />
        </TouchableOpacity>

        {/* Prompt Editor */}
        <View style={[styles.card, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.cardTitle, { color: colors.foreground }]}>Prompt</Text>
          <TextInput
            value={appState.promptText}
            onChangeText={appState.setPromptText}
            placeholder="Enter your prompt here..."
            placeholderTextColor={colors.muted}
            multiline
            numberOfLines={6}
            style={[styles.textArea, { color: colors.foreground, borderColor: colors.border }]}
          />
        </View>

        {/* System Prompt Editor */}
        <View style={[styles.card, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.cardTitle, { color: colors.foreground }]}>System Prompt</Text>
          <TextInput
            value={appState.template.system_prompt}
            onChangeText={(text) => appState.updateTemplate({ system_prompt: text })}
            placeholder="You are a helpful assistant..."
            placeholderTextColor={colors.muted}
            multiline
            numberOfLines={3}
            style={[styles.textArea, { color: colors.foreground, borderColor: colors.border }]}
          />
        </View>

        {/* Personality & Role */}
        <View style={styles.row}>
          <View style={[styles.cardHalf, { borderColor: colors.border, backgroundColor: colors.surface }]}>
            <Text style={[styles.cardLabel, { color: colors.muted }]}>Personality</Text>
            <TextInput
              value={appState.template.personality}
              onChangeText={(text) => appState.updateTemplate({ personality: text })}
              placeholder="helpful"
              placeholderTextColor={colors.muted}
              style={[styles.inputSmall, { color: colors.foreground, borderColor: colors.border }]}
            />
          </View>
          <View style={[styles.cardHalf, { borderColor: colors.border, backgroundColor: colors.surface }]}>
            <Text style={[styles.cardLabel, { color: colors.muted }]}>Role</Text>
            <TextInput
              value={appState.template.role}
              onChangeText={(text) => appState.updateTemplate({ role: text })}
              placeholder="assistant"
              placeholderTextColor={colors.muted}
              style={[styles.inputSmall, { color: colors.foreground, borderColor: colors.border }]}
            />
          </View>
        </View>

        {/* Skill.md Editor */}
        <View style={[styles.card, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.cardTitle, { color: colors.foreground }]}>Skill.md Content</Text>
          <TextInput
            value={appState.template.skill_md_content}
            onChangeText={(text) => appState.updateTemplate({ skill_md_content: text })}
            placeholder="Add skill.md content for model embedding..."
            placeholderTextColor={colors.muted}
            multiline
            numberOfLines={4}
            style={[styles.textArea, { color: colors.foreground, borderColor: colors.border }]}
          />
        </View>

        {/* TTS Controls */}
        <View style={[styles.ttsSection, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <Text style={[styles.cardTitle, { color: colors.foreground }]}>Text-to-Speech Preview</Text>
          <Text style={[styles.ttsHint, { color: colors.muted }]}>
            Preview how your prompt will sound when read aloud
          </Text>
          <TouchableOpacity
            onPress={handleToggleTTS}
            style={[styles.ttsButton, { backgroundColor: isPlaying ? colors.error : colors.primary }]}
          >
            <Ionicons
              name={isPlaying ? "stop-circle-outline" : "play-circle-outline"}
              size={32}
              color="#fff"
            />
            <Text style={styles.ttsButtonText}>
              {isPlaying ? "Stop" : "Play TTS"}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Preview Button */}
        <TouchableOpacity
          onPress={() => setShowPreview(true)}
          style={[styles.previewButton, { backgroundColor: colors.primary }]}
        >
          <Ionicons name="eye-outline" size={20} color="#fff" />
          <Text style={styles.previewButtonText}>Preview Full Prompt</Text>
        </TouchableOpacity>

        {/* Spacer */}
        <View style={{ height: 24 }} />
      </ScrollView>

      {/* Profile Picker Modal */}
      <Modal visible={showProfilePicker} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.foreground }]}>Select Profile</Text>
              <TouchableOpacity onPress={() => setShowProfilePicker(false)}>
                <Ionicons name="close" size={24} color={colors.muted} />
              </TouchableOpacity>
            </View>

            {/* Built-in Presets */}
            <Text style={[styles.modalSectionTitle, { color: colors.muted }]}>BUILT-IN PRESETS</Text>
            {Object.keys(BUILTIN_PRESETS).map((name) => (
              <TouchableOpacity
                key={name}
                onPress={() => applyPreset(name)}
                style={[styles.modalItem, { borderColor: colors.border }]}
              >
                <Ionicons name="flash" size={18} color={colors.primary} />
                <Text style={[styles.modalItemText, { color: colors.foreground }]}>
                  {name.replace("_", " ")}
                </Text>
              </TouchableOpacity>
            ))}

            {/* Saved Profiles */}
            <Text style={[styles.modalSectionTitle, { color: colors.muted }]}>SAVED PROFILES</Text>
            {appState.profiles.map((profile) => (
              <TouchableOpacity
                key={profile.id}
                onPress={() => {
                  appState.setActiveProfile(profile.id);
                  setShowProfilePicker(false);
                }}
                style={[styles.modalItem, { borderColor: colors.border }]}
              >
                <Ionicons
                  name={profile.id === appState.activeProfileId ? "checkmark-circle" : "ellipse-outline"}
                  size={18}
                  color={profile.id === appState.activeProfileId ? colors.primary : colors.muted}
                />
                <Text style={[styles.modalItemText, { color: colors.foreground }]}>{profile.name}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </Modal>

      {/* Template Picker Modal */}
      <Modal visible={showTemplatePicker} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.foreground }]}>Chat Template</Text>
              <TouchableOpacity onPress={() => setShowTemplatePicker(false)}>
                <Ionicons name="close" size={24} color={colors.muted} />
              </TouchableOpacity>
            </View>
            {CHAT_TEMPLATES.map((template) => (
              <TouchableOpacity
                key={template}
                onPress={() => {
                  appState.updateTemplate({ chat_template: template });
                  setShowTemplatePicker(false);
                }}
                style={[styles.modalItem, { borderColor: colors.border }]}
              >
                <Ionicons
                  name={appState.template.chat_template === template ? "checkmark-circle" : "ellipse-outline"}
                  size={18}
                  color={appState.template.chat_template === template ? colors.primary : colors.muted}
                />
                <Text style={[styles.modalItemText, { color: colors.foreground }]}>{template}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </Modal>

      {/* Preview Modal */}
      <Modal visible={showPreview} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.foreground }]}>Full Prompt Preview</Text>
              <TouchableOpacity onPress={() => setShowPreview(false)}>
                <Ionicons name="close" size={24} color={colors.muted} />
              </TouchableOpacity>
            </View>
            <ScrollView style={{ maxHeight: 500, paddingHorizontal: 16, paddingBottom: 16 }}>
              <View style={[styles.previewBlock, { backgroundColor: colors.background }]}>
                <Text style={{ color: colors.foreground, fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace" }}>
                  {fullPrompt || "No content configured yet."}
                </Text>
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
  selector: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 12,
  },
  selectorText: { flex: 1, fontSize: 14, fontWeight: "500" },
  card: {
    borderWidth: 1,
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },
  cardTitle: { fontSize: 14, fontWeight: "600", marginBottom: 8 },
  cardLabel: { fontSize: 12, marginBottom: 4 },
  cardHalf: {
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
    flex: 1,
  },
  row: { flexDirection: "row", gap: 12, marginBottom: 12 },
  textArea: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    lineHeight: 22,
    minHeight: 120,
    textAlignVertical: "top",
  },
  inputSmall: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
  },
  ttsSection: {
    borderWidth: 1,
    borderRadius: 12,
    padding: 16,
    alignItems: "center",
    marginBottom: 12,
  },
  ttsHint: { fontSize: 12, marginTop: 4, marginBottom: 14, textAlign: "center" },
  ttsButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 24,
  },
  ttsButtonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  previewButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 14,
    borderRadius: 12,
  },
  previewButtonText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "flex-end" },
  modalContent: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderWidth: 1,
    paddingBottom: 32,
    maxHeight: "80%",
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
  modalSectionTitle: {
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 1.2,
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 4,
  },
  modalItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: 0.5,
  },
  modalItemText: { fontSize: 15, fontWeight: "500" },
  previewBlock: {
    borderRadius: 8,
    padding: 14,
    fontSize: 13,
    lineHeight: 20,
  },
});
