import { Button, Card, CardBody, Center, Container, Heading, HStack, MultiAutocomplete, Text, useLoading, useSafeLayoutEffect } from "@yamada-ui/react"
import { useState } from "react"
import '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { reduceTo2D } from "./utils";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { PlayIcon } from "@yamada-ui/lucide"

function App() {
  const [value, setValue] = useState<string[]>([])
  const [model, setModel] = useState<use.UniversalSentenceEncoder | null>(null);
  const [reducedEmbeddings, setReducedEmbeddings] = useState<ReturnType<typeof reduceTo2D>>([]);
  const { page } = useLoading()
  const generateEmbedding = async () => {
    if (model && value.length > 0) {
      page.start()
      const embeddingsResult = await model.embed(value);
      const newEmbeddings = await embeddingsResult.array();
      const reduced = reduceTo2D(newEmbeddings, value)
      setReducedEmbeddings(reduced)
      page.finish()
    }
  };

  useSafeLayoutEffect(() => {
    // モデルのロード
    use.load().then(loadedModel => {
      setModel(loadedModel);
    });
  }, []);

  return (
    <Container w="full" as={Center}>
      <Heading>キーワード入力</Heading>
      <HStack w="full" maxW="lg">
        <MultiAutocomplete value={value} onChange={setValue} allowCreate />
        <Button onClick={generateEmbedding} leftIcon={<PlayIcon />}>実行</Button>
      </HStack>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart
          margin={{
            top: 20,
            right: 20,
            bottom: 20,
            left: 20,
          }}
        >
          <CartesianGrid />
          <XAxis type="number" dataKey="x" name="x" />
          <YAxis type="number" dataKey="y" name="y" />
          <Tooltip content={({ active, payload }) => 
            active && payload && payload.length ? (
              <Card bg="white">
                <CardBody>
                  <Text>{payload[0].payload.label}</Text>
                  <Text>X: {payload[0].payload.x.toFixed(4)}</Text>
                  <Text>Y: {payload[0].payload.y.toFixed(4)}</Text>
                </CardBody>
              </Card>
            ) : undefined
          } />
          <Scatter name="Words" data={reducedEmbeddings} fill="#8884d8" />
        </ScatterChart>
      </ResponsiveContainer>
    </Container>
  )
}

export default App