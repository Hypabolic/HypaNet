import { motion } from "framer-motion";
import * as Card from "@/components/card";

export function Testimonial() {
  return (
    <motion.section initial={{ opacity: 0 }}>
      <Card.Root>
        <Card.Title>“It saved our Saturday market”</Card.Title>
        <motion.p className="text-muted-foreground">Stallholder, Riverside Market</motion.p>
      </Card.Root>
    </motion.section>
  );
}
