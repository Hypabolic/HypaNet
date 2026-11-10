import SwiftUI

struct MapControls: View {
    var body: some View {
        HStack {
            RoundIconButton(systemImage: "location", label: "Centre on my boat") {}
            DockButton(systemImage: "anchor", label: "Drop anchor here") {}
            DockButton("Weigh anchor", systemImage: "arrow.up") {}
        }
    }
}
