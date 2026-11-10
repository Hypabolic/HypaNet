import SwiftUI

struct LoomView: View {
    @State private var rows = 12

    var body: some View {
        VStack {
            Text("Warp threads")
                .font(.title2)
            Text("\(rows) rows")
                .accessibilityHidden(true)
            Text("Shuttle parked")
                .hidden()
            Text("Tension ok")
                .opacity(0.4)
            Button("Weave row") { rows += 1 }
        }
        .background {
            Group {
                Button("Undo row") { rows -= 1 }
                    .keyboardShortcut("z", modifiers: .command)
                Button("Next row") { rows += 1 }
                    .keyboardShortcut(.return)
            }
            .opacity(0)
        }
        .overlay {
            Button("Reset loom") { rows = 0 }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .accessibilityHidden(true)
        }
    }
}
