import SwiftUI

struct FeatureList: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Why rangers use it")
                .font(.title2)
                .accessibilityAddTraits(.isHeader)
            FeatureRow(title: "Offline maps", subtitle: "Download a park before you lose signal.")
            FeatureRow(title: "Wildlife log", subtitle: "Pin sightings with a photo and a time.")
            FeatureRow(title: "Shift handover", subtitle: "Leave notes for the next ranger on duty.")
        }
    }

    func footer() -> some View {
        Text("Free for volunteer groups")
            .font(.footnote)
    }
}

struct FeatureRow: View {
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading) {
            Text(title).font(.headline)
            Text(subtitle).foregroundStyle(.secondary)
        }
    }
}

#Preview {
    FeatureList()
}

struct FeatureList_Previews: PreviewProvider {
    static var previews: some View {
        Text("Preview only")
    }
}
