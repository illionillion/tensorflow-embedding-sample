import * as tf from '@tensorflow/tfjs';

export function cosineSimilarity(vec1: number[], vec2: number[]): number {
  const dotProduct = vec1.reduce((sum, a, i) => sum + a * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((sum, a) => sum + a * a, 0));
  const mag2 = Math.sqrt(vec2.reduce((sum, a) => sum + a * a, 0));
  return dotProduct / (mag1 * mag2);
}

export function findSimilarWords(embeddings: number[][], value: string[]) {
  const similarities: { pair: [string, string]; similarity: number }[] = [];

  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      const similarity = cosineSimilarity(embeddings[i], embeddings[j]);
      similarities.push({ pair: [value[i], value[j]], similarity });
    }
  }

  // 類似度でソート
  similarities.sort((a, b) => b.similarity - a.similarity);

  return similarities.map(
    (item) => `${item.pair[0]} - ${item.pair[1]}: ${item.similarity.toFixed(4)}`
  );
}

export function reduceTo2D(embeddings: number[][], labels: string[]): { x: number; y: number; label: string }[] {
  const tensor = tf.tensor2d(embeddings);
  const mean = tensor.mean(0);
  const centered = tensor.sub(mean);

  // QR分解を使用
  const [q, r] = tf.linalg.qr(centered.transpose());

  // 最初の2つの列を取得
  const principalComponents = q.slice([0, 0], [-1, 2]);

  // データを新しい空間に投影
  const reduced = tf.matMul(centered, principalComponents);

  const reducedArray = reduced.arraySync() as number[][];

  // メモリ解放
  tensor.dispose();
  mean.dispose();
  centered.dispose();
  q.dispose();
  r.dispose();
  principalComponents.dispose();
  reduced.dispose();

  // {x: number, y: number, label: string}[]の形式に変換
  return reducedArray.map((point, index) => ({
    x: point[0],
    y: point[1],
    label: labels[index]
  }));
}