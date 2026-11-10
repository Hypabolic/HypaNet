import SwiftUI

struct Step: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
    let seconds: Int
}

struct BrewGuide: View {
    private let steps = [
        Step(title: "Warm the cup", detail: "A splash of hot water, then tip it out", seconds: 10),
        Step(title: "Bloom the grounds", detail: "Wet them and wait", seconds: 30),
        Step(title: "Pour in circles", detail: "Slowly, from the centre outwards", seconds: 90),
    ]
    let grinds: [String]
    static let warnings = [
        String(localized: "Water just off the boil, not boiling"),
        String(localized: "Rinse the paper filter first"),
    ]

    var body: some View {
        List {
            Section("Method") {
                ForEach(steps) { step in
                    VStack(alignment: .leading) {
                        Text(step.title)
                            .font(.headline)
                        Text("\(step.detail) for \(step.seconds) s")
                            .font(.caption)
                    }
                }
            }
            Section("Roast") {
                ForEach(["Light", "Medium", "Dark"], id: \.self) {
                    Text($0)
                }
            }
            Section("Before you start") {
                ForEach(Self.warnings, id: \.self) { warning in
                    Label(warning, systemImage: "exclamationmark.circle")
                }
            }
            Section("Grind") {
                ForEach(grinds, id: \.self) { grind in
                    Text(grind)
                }
            }
        }
        .navigationTitle("Pour-over")
    }
}
