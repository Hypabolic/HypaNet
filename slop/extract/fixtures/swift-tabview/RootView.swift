import SwiftUI

struct RootView: View {
    var body: some View {
        TabView {
            MapScreen()
                .tabItem {
                    Label("Map", systemImage: "map")
                }
            JournalScreen()
                .tabItem {
                    Label("Journal", systemImage: "book")
                }
        }
    }
}

struct JournalScreen: View {
    var body: some View {
        VStack(alignment: .leading) {
            Text("Journal")
                .font(.title)
            Label("Add an entry", systemImage: "plus")
            Link(destination: URL(string: "https://example.org/help")!) {
                HStack {
                    Text("How journaling works")
                    Image(systemName: "arrow.up.right")
                }
            }
        }
    }
}
