import React, { useState, useCallback } from "react";
import {
  ScrollView,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  Alert,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { ScreenContainer } from "@/components/screen-container";
import { useAppState } from "@/hooks/use-app-state";
import { useColors } from "@/hooks/use-colors";
import { BEHAVIOR_CATEGORIES, BUILTIN_PRESETS } from "@/lib/types";
import { StyleSheet } from "react-native";

function BehaviorSlider({
  category,
  value,
  onChange,
  color,
}: {
  category: string;
  value: number;
  onChange: (v: number) => void;
  color: { foreground: string; muted: string; primary: string; surface: string; border: string; background: string; error: string; success: string };
}) {
  const [editing, setEditing] = useState(false);
  const [tempValue, setTempValue] = useState(value.toString());

  const displayValue = value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);

  const valueColor = value > 0 ? color.success : value < 0 ? color.error : color.muted;
  const barWidth = ((value + 1) / 2) * 100;

  return (
    <View style={styles.sliderRow}>
      <Text style={[styles.categoryName, { color: color.foreground }]}>{category.replace(/_/g, " ")}</Text>

      <View style={styles.sliderTrack}>
        {/* Track background */}
        <View style={[styles.sliderBg, { backgroundColor: color.border }]} />
        {/* Center mark */}
        <View style={styles.sliderCenter} />
        {/* Fill */}
        <View
          style={[
            styles.sliderFill,
            {
              width: `${barWidth}%`,
              backgroundColor: valueColor,
            },
          ]}
        />
        {/* Pressable area */}
        <TouchableOpacity
          style={styles.sliderPressArea}
          onPress={() => {
            const step = 0.1;
            const newVal = Math.round((value >= 0.9 ? -1 : value + step) * 10) / 10;
            onChange(newVal);
          }}
          onLongPress={() => onChange(0)}
          activeOpacity={0.6}
        />
      </View>

      <TouchableOpacity
        onPress={() => {
          setEditing(true);
          setTempValue(value.toString());
        }}
        style={[styles.valueBadge, { borderColor: color.border, backgroundColor: color.surface }]}
      >
        <Text style={[styles.valueText, { color: valueColor }]}>{displayValue}</Text>
      </TouchableOpacity>
    </View>
  );
}

export default function ProfilesScreen() {
  const colors = useColors();
  const appState = useAppState();
  const [newProfileName, setNewProfileName] = useState("");
  const [showSaveInput, setShowSaveInput] = useState(false);

  const activeProfile = appState.profiles.find((p) => p.id === appState.activeProfileId) || appState.profiles[0];

  const handleApplyPreset = (presetName: string) => {
    appState.applyProfilePreset(presetName);
  };

  const handleSaveProfile = () => {
    if (!newProfileName.trim()) {
      Alert.alert("Error", "Please enter a profile name");
      return;
    }
    appState.saveProfile(newProfileName.trim());
    setNewProfileName("");
    setShowSaveInput(false);
    if (Platform.OS !== "web") {
      // Haptic feedback
      try {
        const Haptics = require("expo-haptics");
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } catch {}
    }
  };

  const handleDeleteProfile = (id: string) => {
    if (id === activeProfile.id) {
      Alert.alert("Error", "Cannot delete the currently active profile");
      return;
    }
    Alert.alert("Delete Profile", `Are you sure you want to delete "${appState.profiles.find(p => p.id === id)?.name}"?`, [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => appState.deleteProfile(id) },
    ]);
  };

  return (
    <ScreenContainer className="flex-1">
      <ScrollView contentContainerStyle={{ padding: 16 }} style={{ flex: 1 }}>
        {/* Header */}
        <View style={styles.headerRow}>
          <Text style={[styles.headerTitle, { color: colors.foreground }]}>Profiles</Text>
          <Text style={[styles.headerSubtitle, { color: colors.muted }]}>
            Behavior emphasis values
          </Text>
        </View>

        {/* Active Profile Name */}
        <View style={[styles.profileCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
          <View style={styles.profileHeader}>
            <Text style={[styles.profileName, { color: colors.foreground }]}>
              {activeProfile.name}
            </Text>
            <Text style={[styles.profileHint, { color: colors.muted }]}>
              Current profile
            </Text>
          </View>
        </View>

        {/* Presets */}
        <Text style={[styles.sectionTitle, { color: colors.muted }]}>PRESETS</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetScroll}>
          {Object.keys(BUILTIN_PRESETS).map((name) => (
            <TouchableOpacity
              key={name}
              onPress={() => handleApplyPreset(name)}
              style={[styles.presetBtn, { borderColor: colors.border, backgroundColor: colors.surface }]}
            >
              <Ionicons name="flash-outline" size={16} color={colors.primary} />
              <Text style={[styles.presetText, { color: colors.foreground }]}>
                {name.replace("_", " ")}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Behavior Sliders */}
        <Text style={[styles.sectionTitle, { color: colors.muted }]}>BEHAVIOR CATEGORIES</Text>
        {BEHAVIOR_CATEGORIES.map((category) => (
          <BehaviorSlider
            key={category}
            category={category}
            value={activeProfile.values[category]}
            onChange={(v) => appState.updateProfileValue(category, v)}
            color={colors}
          />
        ))}

        {/* Save Profile */}
        <View style={styles.saveSection}>
          {showSaveInput ? (
            <View style={[styles.saveCard, { borderColor: colors.border, backgroundColor: colors.surface }]}>
              <TextInput
                value={newProfileName}
                onChangeText={setNewProfileName}
                placeholder="Profile name..."
                placeholderTextColor={colors.muted}
                style={[styles.saveInput, { color: colors.foreground, borderColor: colors.border }]}
                returnKeyType="done"
                onSubmitEditing={handleSaveProfile}
              />
              <View style={styles.saveActions}>
                <TouchableOpacity
                  onPress={() => {
                    setShowSaveInput(false);
                    setNewProfileName("");
                  }}
                  style={[styles.saveActionBtn, { backgroundColor: colors.border }]}
                >
                  <Text style={[styles.saveActionText, { color: colors.foreground }]}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={handleSaveProfile}
                  style={[styles.saveActionBtn, { backgroundColor: colors.primary }]}
                >
                  <Text style={[styles.saveActionText, { color: "#fff" }]}>Save</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <TouchableOpacity
              onPress={() => setShowSaveInput(true)}
              style={[styles.saveBtn, { borderColor: colors.border, backgroundColor: colors.surface }]}
            >
              <Ionicons name="add-circle-outline" size={20} color={colors.primary} />
              <Text style={[styles.saveBtnText, { color: colors.primary }]}>Save as New Profile</Text>
            </TouchableOpacity>
          )}

          {/* List of saved profiles */}
          {appState.profiles.length > 1 && (
            <View style={styles.savedProfiles}>
              <Text style={[styles.sectionTitle, { color: colors.muted }]}>SAVED PROFILES</Text>
              {appState.profiles
                .filter((p) => p.id !== activeProfile.id)
                .map((profile) => (
                  <TouchableOpacity
                    key={profile.id}
                    onPress={() => appState.setActiveProfile(profile.id)}
                    style={[styles.savedProfileItem, { borderColor: colors.border, backgroundColor: colors.surface }]}
                  >
                    <View style={{ flex: 1, flexDirection: "row", alignItems: "center", gap: 8 }}>
                      <Ionicons name="person-outline" size={16} color={colors.muted} />
                      <Text style={[styles.savedProfileName, { color: colors.foreground }]}>
                        {profile.name}
                      </Text>
                    </View>
                    <TouchableOpacity
                      onPress={() => handleDeleteProfile(profile.id)}
                      hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                      <Ionicons name="trash-outline" size={16} color={colors.error} />
                    </TouchableOpacity>
                  </TouchableOpacity>
                ))}
            </View>
          )}
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
  profileCard: {
    borderWidth: 1,
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
  },
  profileHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  profileName: { fontSize: 18, fontWeight: "700" },
  profileHint: { fontSize: 12 },
  sectionTitle: {
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 1.2,
    marginTop: 16,
    marginBottom: 10,
  },
  presetScroll: { marginBottom: 16 },
  presetBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    borderWidth: 1,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 8,
    marginRight: 10,
  },
  presetText: { fontSize: 13, fontWeight: "600" },
  sliderRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 10,
  },
  categoryName: {
    fontSize: 13,
    fontWeight: "500",
    width: 100,
    textTransform: "capitalize",
  },
  sliderTrack: {
    flex: 1,
    height: 28,
    borderRadius: 14,
    justifyContent: "center",
    position: "relative",
    overflow: "hidden",
  },
  sliderBg: {
    position: "absolute",
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    borderRadius: 14,
    height: 6,
    alignSelf: "center",
    width: "100%",
  },
  sliderCenter: {
    position: "absolute",
    left: "50%",
    top: 4,
    bottom: 4,
    width: 2,
    backgroundColor: "#888",
    opacity: 0.4,
  },
  sliderFill: {
    position: "absolute",
    top: 4,
    height: 6,
    borderRadius: 3,
    opacity: 0.8,
  },
  sliderPressArea: {
    position: "absolute",
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  valueBadge: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    minWidth: 52,
    alignItems: "center",
  },
  valueText: { fontSize: 13, fontWeight: "700", fontVariant: ["tabular-nums"] },
  saveSection: { marginTop: 16 },
  saveCard: {
    borderWidth: 1,
    borderRadius: 14,
    padding: 14,
  },
  saveInput: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    marginBottom: 10,
  },
  saveActions: { flexDirection: "row", justifyContent: "flex-end", gap: 8 },
  saveActionBtn: { paddingVertical: 8, paddingHorizontal: 18, borderRadius: 8 },
  saveActionText: { fontSize: 13, fontWeight: "600" },
  saveBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    justifyContent: "center",
  },
  saveBtnText: { fontSize: 14, fontWeight: "600" },
  savedProfiles: { marginTop: 16 },
  savedProfileItem: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderWidth: 1,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    marginBottom: 8,
  },
  savedProfileName: { fontSize: 14, fontWeight: "500" },
});
