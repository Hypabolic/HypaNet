import SwiftUI

struct PauseOverlay: View {
    var onResume: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            VStack(spacing: 8) {
                Text("Expedition halted")
                    .font(.caption)
                    .textCase(.uppercase)
                    .tracking(3)
                Text("PAUSED")
                    .font(.largeTitle)
                Text("Resume returns control to the rover.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            VStack(spacing: 12) {
                Button("RESUME", action: onResume)
                Button("OPTIONS") {}
                Button("ABANDON RUN", role: .destructive) {}
            }
        }
        .padding(40)
    }
}
