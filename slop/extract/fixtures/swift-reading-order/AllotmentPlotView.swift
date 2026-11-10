import SwiftUI

struct AllotmentPlotView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading) {
                header
                if showsRota {
                    rotaSection
                }
                self.notesSection
                    .padding()
            }
        }
        .toolbar { toolbarItems }
    }

    private var showsRota: Bool { true }

    private var notesSection: some View {
        VStack {
            Text("Notes").font(.headline)
            Text("Mulch the beans before the dry spell.")
        }
    }

    @ToolbarContentBuilder
    private var toolbarItems: some ToolbarContent {
        ToolbarItem { Button("Share plot") {} }
    }

    private var rotaSection: some View {
        VStack {
            Text("Watering rota").font(.headline)
            Text("Plot holders take a week each.")
        }
    }

    private var header: some View {
        Text("Plot 14").font(.largeTitle)
    }

    private var unusedFooter: some View {
        Text("Last updated by the committee")
    }
}
