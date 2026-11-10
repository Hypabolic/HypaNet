import SwiftUI

struct Moorings: View {
    @Bindable var model: MooringModel

    var body: some View {
        List(model.berths) { berth in
            Text(berth.name)
        }
        .alert(
            "Berth unavailable",
            isPresented: $model.isShowingClash,
            actions: {
                Button("Pick another") { model.pickAnother() }
                Button("Cancel", role: .cancel) {}
            },
            message: { Text("Another boat has this berth for the night.") }
        )
    }
}
