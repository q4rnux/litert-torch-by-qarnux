import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import { SymbolWeight, SymbolViewProps } from "expo-symbols";
import { ComponentProps } from "react";
import { OpaqueColorValue, type StyleProp, type TextStyle } from "react-native";

type IconMapping = Record<string, ComponentProps<typeof MaterialIcons>["name"]>;
type IconSymbolName = keyof typeof MAPPING;

const MAPPING: IconMapping = {
  "house.fill": "home",
  "paperplane.fill": "send",
  "chevron.left.forwardslash.chevron.right": "code",
  "chevron.right": "chevron-right",
  "mic.fill": "mic",
  "mic.slash.fill": "mic-off",
  "pause.fill": "pause",
  "play.fill": "play-arrow",
  "stop.fill": "stop",
  "waveform.path.ecg.rectangle.fill": "graphic-eq",
  "account.tree": "account-tree",
  "tune": "tune",
  "apps.fill": "apps",
  "gearshape.fill": "settings",
  "list.bullet": "list",
  "plus.circle.fill": "add-circle",
  "trash.fill": "delete",
  "square.and.arrow.up.fill": "share",
  "doc.on.doc.fill": "file-copy",
  "text.alignleft": "format-align-left",
  "person.fill": "person",
  "slider.horizontal.3": "tune",
  "chevron.down": "expand-more",
  "chevron.up": "expand-less",
  "arrow.clockwise": "refresh",
  "exclamationmark.triangle.fill": "warning",
  "checkmark.circle.fill": "check-circle",
  "xmark.circle.fill": "cancel",
  "bolt.fill": "bolt",
  "sparkles": "auto-awesome",
  "brain.head.profile": "psychology",
  "book.fill": "menu-book",
  "rectangle.stack.fill": "layers",
  "square.stack.3d.up.fill": "view-stream",
  "arrow.right": "arrow-forward",
  "arrow.left": "arrow-back",
  "arrowtriangle.forward.fill": "arrow-forward",
  "arrowtriangle.backward.fill": "arrow-back",
  "waveform": "graphic-eq",
  "speaker.wave.2.fill": "volume-up",
  "speaker.slash.fill": "volume-off",
  "doc.plaintext": "description",
  "square.and.arrow.down.fill": "download",
} as IconMapping;

export function IconSymbol({
  name,
  size = 24,
  color,
  style,
}: {
  name: IconSymbolName;
  size?: number;
  color: string | OpaqueColorValue;
  style?: StyleProp<TextStyle>;
  weight?: SymbolWeight;
}) {
  return <MaterialIcons color={color} size={size} name={MAPPING[name]} style={style} />;
}
