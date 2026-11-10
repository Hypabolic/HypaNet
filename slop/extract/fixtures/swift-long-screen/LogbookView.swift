import SwiftUI

struct LogbookView: View {
    var body: some View {
        List {
            Section("Boat") {
                LabeledContent("Hull number", value: "Not set")
                LabeledContent("Home harbour", value: "Not set")
                LabeledContent("Length overall", value: "Not set")
                LabeledContent("Draught", value: "Not set")
            }
            Section("Engine") {
                LabeledContent("Hours run", value: "Not set")
                LabeledContent("Last oil change", value: "Not set")
                LabeledContent("Impeller replaced", value: "Not set")
                LabeledContent("Fuel type", value: "Not set")
            }
            Section("Safety") {
                LabeledContent("Life jackets", value: "Not set")
                LabeledContent("Flares expiry", value: "Not set")
                LabeledContent("Fire extinguisher", value: "Not set")
                LabeledContent("First aid kit", value: "Not set")
            }
            Section("Radio") {
                LabeledContent("Call sign", value: "Not set")
                LabeledContent("MMSI", value: "Not set")
                LabeledContent("Licence expiry", value: "Not set")
                LabeledContent("Handheld spare", value: "Not set")
            }
            Section("Crew") {
                LabeledContent("Skipper", value: "Not set")
                LabeledContent("First mate", value: "Not set")
                LabeledContent("Emergency contact", value: "Not set")
                LabeledContent("Crew limit", value: "Not set")
            }
            Section("Mooring") {
                LabeledContent("Berth", value: "Not set")
                LabeledContent("Annual fee", value: "Not set")
                LabeledContent("Electric hook-up", value: "Not set")
                LabeledContent("Water tap", value: "Not set")
            }
            Section("Insurance") {
                LabeledContent("Insurer", value: "Not set")
                LabeledContent("Policy number", value: "Not set")
                LabeledContent("Renewal date", value: "Not set")
                LabeledContent("Excess", value: "Not set")
            }
            Section("Sails") {
                LabeledContent("Mainsail age", value: "Not set")
                LabeledContent("Genoa age", value: "Not set")
                LabeledContent("Spinnaker", value: "Not set")
                LabeledContent("Reefs", value: "Not set")
            }
        }
        .navigationTitle("Logbook")
    }
}
