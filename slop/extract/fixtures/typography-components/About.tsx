import { Typography } from "@mui/material";
import { Heading, Text } from "@chakra-ui/react";

export function About() {
  return (
    <div>
      <Typography variant="overline">Our story</Typography>
      <Heading as="h2">Two benches and a kiln</Heading>
      <Text>We started firing mugs in a garage in 2019.</Text>
      <Typography variant="caption">Photos by a member of the studio</Typography>
      <Heading level={5}>Visiting</Heading>
      <Typography variant="h6">Opening hours</Typography>
    </div>
  );
}
