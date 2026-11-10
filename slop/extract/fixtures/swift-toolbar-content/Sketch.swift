import SwiftUI

struct Sketch: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Canvas { _, _ in }
                .navigationTitle("Sketch")
                .toolbar { bar }
        }
    }

    @ToolbarContentBuilder private var bar: some ToolbarContent {
        ToolbarItem(placement: .cancellationAction) {
            Button("Discard") { dismiss() }
        }
        ToolbarItem(placement: .confirmationAction) {
            Button("Keep sketch") { dismiss() }
        }
    }
}
