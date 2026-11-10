import SwiftUI

struct BrewView: View {
    @State private var cups = 2

    var body: some View {
        ScrollView {
            GroupBox("Today's brew") {
                Stepper("Cups: \(cups)", value: $cups, in: 1...8)
                LabeledContent("Grind", value: "Medium-fine")
            }
            Menu("Brew method") {
                Button("Pour over") {}
                Button("French press") {}
            }
            DisclosureGroup("Water notes") {
                Text("Soft water brings out the fruit in lighter roasts.")
            }
        }
    }
}
