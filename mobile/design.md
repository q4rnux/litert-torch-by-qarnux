# TTS Prompter — Mobile App Interface Design

## Brand Identity

- **Brand**: qarnux
- **Color Palette**:
  - Primary: Deep navy blue `#1B2A4A` (light) / `#0F1A30` (dark)
  - Accent: Electric blue `#3B82F6`
  - Surface: Slate `#F1F5F9` (light) / `#1E293B` (dark)
  - Text: `#0F172A` (light) / `#E2E8F0` (dark)
  - Muted: `#64748B` (light) / `#94A3B8` (dark)
  - Border: `#CBD5E1` (light) / `#334155` (dark)
  - Success: `#10B981`
  - Warning: `#F59E0B`
  - Error: `#EF4444`

## Screen List

### 1. Prompter (Home Tab)
- **Primary Content**: TTS prompt preview area with text input/editing
- **Functionality**: Write/edit prompts, hear them via TTS, select behavior profile + template, preview full system prompt
- **Key Elements**: Large text area for prompt editing, TTS play/pause/stop buttons, profile selector, template selector, preview button

### 2. Pipeline Tab
- **Primary Content**: Animated 7-stage orchestration pipeline visualization
- **Functionality**: Shows the agent flow with status indicators, tap to see agent details
- **Key Elements**: Horizontal/vertical flow diagram with 7 agent nodes, connection lines, status badges, expandable agent detail cards

### 3. Profiles Tab
- **Primary Content**: Behavior profile editor with sliders
- **Functionality**: Adjust emphasis values (-1.0 to 1.0) for 15 behavior categories, apply presets, save/load profiles
- **Key Elements**: Slider for each category, preset buttons at top, save/export buttons

### 4. Models Tab
- **Primary Content**: Model architecture library browser
- **Functionality**: Browse supported architectures, view prompt templates, configure quantization settings
- **Key Elements**: Card list of architectures (Llama, Gemma, Mistral, Qwen, Phi), detail view with template preview, quantization config panel

### 5. Settings Tab
- **Primary Content**: App settings and export configuration
- **Functionality**: Export profiles/templates as JSON/YAML, manage saved configurations, app preferences
- **Key Elements**: Export buttons, saved configs list, about section, theme toggle

## Key User Flows

### Flow 1: Craft a Prompt with TTS Preview
1. User opens Prompter tab (default)
2. User selects a behavior profile (e.g., "coding_expert") from dropdown
3. User selects a chat template (e.g., "llama3")
4. User types/edits their prompt in the text area
5. User taps "Preview Prompt" to see the full assembled system prompt
6. User taps "Play TTS" to hear the prompt read aloud
7. User adjusts settings and re-previews

### Flow 2: Create a Behavior Profile
1. User navigates to Profiles tab
2. User taps a preset (e.g., "creative_writer")
3. User adjusts individual sliders (creativity, humor, empathy, etc.)
4. User taps "Save Profile" and gives it a name
5. User taps "Export" to generate JSON/YAML config

### Flow 3: Configure Quantization
1. User navigates to Models tab
2. User selects an architecture (e.g., "Llama")
3. User views the prompt template
4. User configures quantization settings (recipe, method, dtype, etc.)
5. User exports the configuration

### Flow 4: View Pipeline
1. User navigates to Pipeline tab
2. User sees animated flow from ParserAgent to PackagingAgent
3. User taps an agent node to see details about that stage

## Tab Bar Navigation

| Tab | Icon | Name |
|-----|------|------|
| Prompter | mic | Prompter |
| Pipeline | account-tree | Pipeline |
| Profiles | tune | Profiles |
| Models | apps | Models |
| Settings | settings | Settings |
