import SwiftUI

struct KettleTimerView: View {
    var body: some View {
        List {
            Section("Brew") {
                Text("Steep for four minutes")
                #if os(iOS)
                Text("Tap the lid to start")
                #else
                Text("Press space to start")
                #endif
            }
        }
    }
}
