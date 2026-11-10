import { Hero } from "@/components/hero";
import { FeatureTile } from "@/components/feature-tile";

export function LandingHero() {
  return (
    <>
      <Hero
        eyebrow="For school allotments"
        title="Grow lunch with your class"
        subtitle="Lesson plans that follow the seasons, from sowing to soup."
        cta="Download the term planner"
      />
      <FeatureTile
        title={<span>Weekly <em>jobs</em></span>}
        description="A short list for each week of term."
        badge="Free"
      />
    </>
  );
}
