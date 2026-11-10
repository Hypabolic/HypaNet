import SwiftUI

struct HarbourList: View {
    let moorings: [String]
    let filter: String

    var body: some View {
        VStack {
            if moorings.isEmpty {
                Text("No boats moored")
                    .font(.headline)
                Text("Moorings you book appear here with their tide times.")
                    .font(.callout)
            } else {
                ForEach(moorings, id: \.self) { m in
                    Text(m)
                }
            }
            if filter.isEmpty {
                Text(filter)
            }
        }
    }
}
