import SwiftUI

struct Units: View {
    @State private var unit = 0
    @State private var loud = true

    var body: some View {
        Form {
            Section("Units") {
                Picker("Distance unit", selection: $unit) {
                    Text("Nautical miles").tag(0)
                    Text("Kilometres").tag(1)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Toggle("Loud fog horn", isOn: $loud)
            }
        }
    }
}
