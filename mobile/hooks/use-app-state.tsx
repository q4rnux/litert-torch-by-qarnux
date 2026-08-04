import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  BehaviorProfile,
  TemplateConfig,
  QuantizationConfig,
  createDefaultProfile,
  createDefaultTemplate,
  createDefaultQuantizationConfig,
  applyPreset,
  BUILTIN_PRESETS,
} from "@/lib/types";

interface AppState {
  profiles: BehaviorProfile[];
  activeProfileId: string;
  template: TemplateConfig;
  quantization: QuantizationConfig;
  savedConfigs: { name: string; data: string; format: "json" | "yaml" }[];
  promptText: string;
}

interface AppStateContextType extends AppState {
  // Profile actions
  setActiveProfile: (id: string) => void;
  updateProfileValue: (category: string, value: number) => void;
  applyProfilePreset: (presetName: string) => void;
  saveProfile: (name: string) => void;
  deleteProfile: (id: string) => void;

  // Template actions
  updateTemplate: (updates: Partial<TemplateConfig>) => void;

  // Quantization actions
  updateQuantization: (updates: Partial<QuantizationConfig>) => void;

  // Prompt actions
  setPromptText: (text: string) => void;

  // Config export
  exportConfig: (name: string, format: "json" | "yaml") => string;
  deleteConfig: (index: number) => void;
}

const AppStateContext = createContext<AppStateContextType | null>(null);

const STORAGE_KEY = "tts_prompter_state";

function getDefaultState(): AppState {
  const defaultProfile = createDefaultProfile("default");
  return {
    profiles: [defaultProfile],
    activeProfileId: defaultProfile.id,
    template: createDefaultTemplate(),
    quantization: createDefaultQuantizationConfig(),
    savedConfigs: [],
    promptText: "",
  };
}

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AppState>(getDefaultState);
  const [loaded, setLoaded] = useState(false);

  // Load from AsyncStorage on mount
  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY)
      .then((data) => {
        if (data) {
          try {
            const parsed = JSON.parse(data) as AppState;
            setState(parsed);
          } catch {
            // Use defaults
          }
        }
      })
      .finally(() => setLoaded(true));
  }, []);

  // Save to AsyncStorage on state change
  useEffect(() => {
    if (loaded) {
      AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }
  }, [state, loaded]);

  const activeProfile = state.profiles.find((p) => p.id === state.activeProfileId) || state.profiles[0];

  const setActiveProfile = useCallback((id: string) => {
    setState((prev) => ({ ...prev, activeProfileId: id }));
  }, []);

  const updateProfileValue = useCallback((category: string, value: number) => {
    setState((prev) => ({
      ...prev,
      profiles: prev.profiles.map((p) =>
        p.id === prev.activeProfileId
          ? { ...p, values: { ...p.values, [category]: value } }
          : p
      ),
    }));
  }, []);

  const applyProfilePreset = useCallback((presetName: string) => {
    setState((prev) => ({
      ...prev,
      profiles: prev.profiles.map((p) =>
        p.id === prev.activeProfileId ? applyPreset(p, presetName) : p
      ),
    }));
  }, []);

  const saveProfile = useCallback((name: string) => {
    setState((prev) => {
      const existing = prev.profiles.find(
        (p) => p.name === name && p.id !== prev.activeProfileId
      );
      if (existing) return prev;
      const newProfile = {
        ...activeProfile,
        id: `profile_${Date.now()}`,
        name,
      };
      return {
        ...prev,
        profiles: [...prev.profiles, newProfile],
        activeProfileId: newProfile.id,
      };
    });
  }, [activeProfile]);

  const deleteProfile = useCallback((id: string) => {
    setState((prev) => {
      const newProfiles = prev.profiles.filter((p) => p.id !== id);
      if (newProfiles.length === 0) {
        const def = createDefaultProfile("default");
        return { ...prev, profiles: [def], activeProfileId: def.id };
      }
      return {
        ...prev,
        profiles: newProfiles,
        activeProfileId:
          prev.activeProfileId === id ? newProfiles[0].id : prev.activeProfileId,
      };
    });
  }, []);

  const updateTemplate = useCallback((updates: Partial<TemplateConfig>) => {
    setState((prev) => ({
      ...prev,
      template: { ...prev.template, ...updates },
    }));
  }, []);

  const updateQuantization = useCallback((updates: Partial<QuantizationConfig>) => {
    setState((prev) => ({
      ...prev,
      quantization: { ...prev.quantization, ...updates },
    }));
  }, []);

  const setPromptText = useCallback((text: string) => {
    setState((prev) => ({ ...prev, promptText: text }));
  }, []);

  const objectToYaml = (obj: any, indent = 0): string => {
    const pad = "  ".repeat(indent);
    if (obj === null || obj === undefined) return `${pad}null`;
    if (typeof obj === "number" || typeof obj === "boolean") return `${pad}${obj}`;
    if (typeof obj === "string") {
      if (obj.includes("\n") || obj.includes(":") || obj.includes("#")) {
        return `${pad}|\n${obj.split("\n").map((l: string) => `${pad}  ${l}`).join("\n")}`;
      }
      return `${pad}"${obj.replace(/"/g, '\\"')}"`;
    }
    if (Array.isArray(obj)) {
      if (obj.length === 0) return `${pad}[]`;
      return obj.map((item) => {
        if (typeof item === "object" && item !== null) {
          const firstKey = Object.keys(item)[0];
          const val = objectToYaml(item[firstKey], indent + 1).trim();
          return `${pad}- ${firstKey}: ${val}`;
        }
        return `${pad}- ${objectToYaml(item, indent + 1).trim()}`;
      }).join("\n");
    }
    if (typeof obj === "object") {
      const entries = Object.entries(obj);
      if (entries.length === 0) return `${pad}{}`;
      return entries
        .map(([key, val]) => {
          const yamlVal = objectToYaml(val, indent + 1);
          return `${pad}${key}:${yamlVal.startsWith("\n") ? yamlVal : " " + yamlVal.trim()}`;
        })
        .join("\n");
    }
    return `${pad}${obj}`;
  };

  const exportConfig = useCallback(
    (name: string, format: "json" | "yaml"): string => {
      const config = {
        behavior_profile: activeProfile,
        template: state.template,
        quantization: state.quantization,
        prompt: state.promptText,
      };

      let data: string;
      if (format === "json") {
        data = JSON.stringify(config, null, 2);
      } else {
        data = objectToYaml(config);
      }

      setState((prev) => ({
        ...prev,
        savedConfigs: [
          ...prev.savedConfigs,
          { name, data, format },
        ],
      }));

      return data;
    },
    [activeProfile, state.template, state.quantization, state.promptText]
  );

  const deleteConfig = useCallback((index: number) => {
    setState((prev) => ({
      ...prev,
      savedConfigs: prev.savedConfigs.filter((_, i) => i !== index),
    }));
  }, []);

  return (
    <AppStateContext.Provider
      value={{
        ...state,
        setActiveProfile,
        updateProfileValue,
        applyProfilePreset,
        saveProfile,
        deleteProfile,
        updateTemplate,
        updateQuantization,
        setPromptText,
        exportConfig,
        deleteConfig,
      }}
    >
      {children}
    </AppStateContext.Provider>
  );
}

export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) throw new Error("useAppState must be used within AppStateProvider");
  return context;
}
