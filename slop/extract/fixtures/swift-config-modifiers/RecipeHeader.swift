import SwiftUI

struct RecipeHeader: View {
    let recipe: Recipe

    var body: some View {
        VStack(alignment: .leading) {
            Text("Weeknight supper")
                .brandFont(.kicker)
            Text(recipe.title)
                .brandFont(.display)
            Text("Ready in under half an hour")
                .brandFont(.note)
        }
    }
}
