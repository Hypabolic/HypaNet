import SwiftUI

struct Logbook: View {
    @State private var page: String?

    var body: some View {
        NavigationSplitView {
            List(selection: $page) {
                Section("Voyages") {
                    Label("Passages", systemImage: "sailboat")
                    Label("Anchorages", systemImage: "anchor")
                }
            }
            .navigationTitle("Logbook")
        } detail: {
            Text("Choose a page from the log.")
        }
    }
}
