import SwiftUI
import PhotosUI

struct GardenToolbar: View {
    @State private var pick: PhotosPickerItem?
    var body: some View {
        VStack {
            Text("Seed packets")
            PhotosPicker(selection: $pick, matching: .images) {
                Label("Add a bed photo", systemImage: "camera")
            }
            PhotosPicker("Choose cover", selection: $pick)
            ShareLink(item: URL(string: "https://example.com")!) {
                Label("Send planting plan", systemImage: "square.and.arrow.up")
            }
            ShareLink("Share harvest", item: "x")
            Menu {
                Button("Rename bed") {}
                Button("Archive bed") {}
            } label: {
                Label("Bed options", systemImage: "ellipsis.circle")
                    .labelStyle(.iconOnly)
            }
            Button {
            } label: {
                Label("Water now", systemImage: "drop")
            }
            .labelStyle(.iconOnly)
            Label("Frost tonight", systemImage: "snowflake")
            EditButton()
            RenameButton()
            PasteButton(payloadType: String.self) { _ in }
            Menu {
                Text("Sort by")
            } label: {
                Text("Sort")
            }
        }
    }
}
