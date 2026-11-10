import SwiftUI

struct SettingsView: View {
    @State private var syncOnCellular = false
    @State private var units = "Metric"
    @State private var nickname = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Sync") {
                    Toggle("Sync over mobile data", isOn: $syncOnCellular)
                        .help("Large photo uploads still wait for Wi-Fi.")
                }
                Section {
                    Picker("Units", selection: $units) {
                        Text("Metric").tag("Metric")
                        Text("Imperial").tag("Imperial")
                    }
                    TextField("Trail nickname", text: $nickname)
                } header: {
                    Text("Display")
                } footer: {
                    Text("Units apply to distances and elevation.")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
