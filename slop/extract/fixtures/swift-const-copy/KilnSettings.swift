import SwiftUI

enum Firing: CaseIterable {
    case bisque, glaze, raku
}

extension Firing {
    var label: String {
        switch self {
        case .bisque: "Bisque"
        case .glaze: "Glaze"
        case .raku: String(localized: "Raku")
        }
    }
}

struct KilnSettings: View {
    let firing: Firing
    let note: String
    @State private var draft = "Untitled load"
    private let heading = "Kiln settings"
    static let footer = String(localized: "Cone values follow Orton charts.")

    var body: some View {
        Form {
            Section {
                Picker("Firing", selection: .constant(firing)) {
                    ForEach(Firing.allCases, id: \.self) { Text($0.label) }
                }
                Text(firing.label)
                Text(status)
                Text(warning(for: firing))
                Text(note)
                Text(draft)
                Text(summary)
            } header: {
                Text(heading)
            } footer: {
                Text(Self.footer)
            }
        }
        .navigationTitle(heading)
    }

    private var status: String {
        if firing == .raku {
            return "Outdoor kiln only"
        }
        return "Ready to load"
    }

    private func warning(for firing: Firing) -> String {
        switch firing {
        case .bisque: "Leave the bung out until 600 °C."
        case .glaze, .raku: "Check shelf wash before loading."
        }
    }

    private var summary: String {
        switch firing {
        case .bisque: "First firing"
        default: note
        }
    }
}

struct LoadRow: View {
    let heading: String

    var body: some View {
        Text(heading)
    }
}
