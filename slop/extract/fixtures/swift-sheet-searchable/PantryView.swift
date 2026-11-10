import SwiftUI

struct PantryView: View {
    @State private var query = ""
    @State private var showAdd = false
    let lowStock: Bool

    var body: some View {
        NavigationStack {
            VStack {
                if lowStock {
                    Text("Three items are running low")
                } else {
                    Text(NSLocalizedString("pantry.allStocked", comment: ""))
                }
                Text(lowStock ? "Restock soon" : "All good")
                Text("Best before ") + Text("Friday").bold()
                Button {
                    showAdd = true
                } label: {
                    Image(systemName: "plus")
                }
                .accessibilityLabel("Add pantry item")
            }
            .searchable(text: $query, prompt: "Search jars and tins")
            .sheet(isPresented: $showAdd) {
                Text("New pantry item")
                    .font(.title2)
                TextField("Name", text: $query, prompt: Text("e.g. Chickpeas"))
            }
        }
    }
}
