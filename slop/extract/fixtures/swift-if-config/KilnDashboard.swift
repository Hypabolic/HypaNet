import SwiftUI

struct KilnDashboard: View {
    var body: some View {
        VStack {
            Text("Bisque firing")
            #if DEBUG
            Button("Skip to cooling") {}
            Text("Thermocouple raw: 0.82 mV")
            #else
            Text("Firing safely")
            #endif
            #if os(iOS)
            Text("Tap a shelf to log it")
            #elseif os(macOS)
            Text("Click a shelf to log it")
            #endif
            #if os(iOS)
            HStack {
                Text("Kiln")
                Text("Shelf map")
            }
            #else
            Text("Open the shelf map")
            #endif
            #if !DEBUG
            Text("Glaze notes are synced")
            #else
            Text("Sync is off in test builds")
            #endif
        }
        #if DEBUG
        .overlay { Text("Preview data") }
        #endif
    }
}
