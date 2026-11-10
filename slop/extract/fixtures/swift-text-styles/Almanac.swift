import SwiftUI

struct Almanac: View {
    let entry: AlmanacEntry

    var body: some View {
        VStack(alignment: .leading) {
            Text(entry.title)
                .brandFont(.title)
            Text("High water")
                .appFont(.headline)
            Text(entry.summary)
                .appFont(.body)
            Text("Times are local, adjusted for summer time.")
                .brandFont(.captionBold)
            Text("Moon phase")
                .brandFont(.display)
            Text("Allowed since \(entry.date, format: .dateTime.day().month())")
        }
    }
}
