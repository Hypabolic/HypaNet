import { Pressable, Text, TextInput, View } from "react-native";

export default function TideScreen({ height }: { height: number }) {
  return (
    <View>
      <Text style={{ fontSize: 28 }}>Tides</Text>
      <Text>
        High water at <Text style={{ fontWeight: "700" }}>14:32</Text> today
      </Text>
      <TextInput placeholder="Search a harbour" />
      <Pressable onPress={() => {}}>
        <Text>Set an alert</Text>
      </Pressable>
      <Text style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: 2 }}>Next low</Text>
      <Text style={{ fontSize: 40 }}>{height}</Text>
    </View>
  );
}
