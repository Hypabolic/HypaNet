import SwiftUI

struct CookSettingsView: View {
    @State private var keepAwake = true
    @State private var voice = false
    @State private var showDelete = false
    @State private var recipes: [String] = []

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Keep screen awake", isOn: $keepAwake)
                    Toggle("Voice commands", isOn: $voice)
                } header: {
                    Text("Cooking")
                } footer: {
                    Text("The screen stays on while a recipe is open.")
                }

                Section {
                    Stepper("Default servings", value: .constant(4))
                } header: {
                    Text("Servings")
                } footer: {
                    Text("Choose the default servings.")
                }

                Section {
                    Button("Delete All Timers", role: .destructive) { showDelete = true }
                } header: {
                    Text("Timers")
                } footer: {
                    Text("Manage your timers.")
                }
            }
            .navigationTitle("Settings")
            .alert("Delete all timers?", isPresented: $showDelete) {
                Button("Delete", role: .destructive) {}
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Running timers stop immediately.")
            }
            .overlay {
                if recipes.isEmpty {
                    ContentUnavailableView {
                        Label("No Recipes", systemImage: "book")
                    } description: {
                        Text("You don't have any recipes yet.")
                    } actions: {
                        Button("Scan a Recipe") {}
                    }
                }
            }
        }
    }
}

struct PausedOverlay: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("Paused")
                .font(.largeTitle)
            Text("Timers are stopped until you resume.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Button("Resume cooking") {}
        }
    }
}

struct LibraryHeader: View {
    var body: some View {
        VStack(alignment: .leading) {
            Text("Library")
                .font(.title)
            Text("All your saved recipes in one place.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
