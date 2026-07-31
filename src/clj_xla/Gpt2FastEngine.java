package clj_xla;

public class Gpt2FastEngine {

    private static double clamp(double d) {
        if (Double.isNaN(d)) return 0.0;
        if (d > 3.4028234663852886E38) return 3.4028234663852886E38;
        if (d < -3.4028234663852886E38) return -3.4028234663852886E38;
        return d;
    }

    public static float[] evalSequence(float[][] X,
                                      float[][] ln1G, float[][] ln1B,
                                      float[][] cAttnW, float[][] cAttnB,
                                      float[][] cProjW, float[][] cProjB,
                                      float[][] ln2G, float[][] ln2B,
                                      float[][] mlpFcW, float[][] mlpFcB,
                                      float[][] mlpProjW, float[][] mlpProjB,
                                      float[] lnFG, float[] lnFB) {
        int S = X.length;
        int embDim = 768;
        int numHeads = 12;
        int headDim = 64;
        int numLayers = ln1G.length;

        float[][] CurrM = new float[S][embDim];
        for (int s = 0; s < S; s++) {
            System.arraycopy(X[s], 0, CurrM[s], 0, embDim);
        }

        double[] qkvAcc = new double[2304];
        double[] projAcc = new double[embDim];
        double[] fcAcc = new double[3072];
        double[] scoresBuf = new double[1024];
        double[] expsBuf = new double[1024];

        for (int l = 0; l < numLayers; l++) {
            float[] g1 = ln1G[l], b1 = ln1B[l];
            float[] cAtW = cAttnW[l], cAtB = cAttnB[l];
            float[] cPrW = cProjW[l], cPrB = cProjB[l];
            float[] g2 = ln2G[l], b2 = ln2B[l];
            float[] fcW = mlpFcW[l], fcB = mlpFcB[l];
            float[] prW = mlpProjW[l], prB = mlpProjB[l];

            // 1. LayerNorm 1
            float[][] MNorm1 = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                float[] r = CurrM[pos];
                double sum = 0.0;
                for (int i = 0; i < embDim; i++) sum += r[i];
                double mean = sum / embDim;
                double varSum = 0.0;
                for (int i = 0; i < embDim; i++) {
                    double d = r[i] - mean;
                    varSum += d * d;
                }
                double std = Math.sqrt((varSum / embDim) + 1e-5);
                for (int i = 0; i < embDim; i++) {
                    MNorm1[pos][i] = (float) clamp(((r[i] - mean) / std) * g1[i] + b1[i]);
                }
            }

            // 2. QKV Projection: r @ cAtW [768 x 2304] + cAtB
            float[][] QKV = new float[S][2304];
            for (int pos = 0; pos < S; pos++) {
                float[] r = MNorm1[pos];
                for (int j = 0; j < 2304; j++) qkvAcc[j] = cAtB[j];
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * 2304;
                    for (int j = 0; j < 2304; j++) {
                        qkvAcc[j] += v * cAtW[rowOff + j];
                    }
                }
                for (int j = 0; j < 2304; j++) {
                    QKV[pos][j] = (float) clamp(qkvAcc[j]);
                }
            }

            // 3. Causal Self-Attention
            float[][] AttnMerged = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                float[] qkvI = QKV[pos];
                for (int h = 0; h < numHeads; h++) {
                    int hOff = h * headDim;
                    int bound = pos + 1;
                    for (int j = 0; j < bound; j++) {
                        float[] qkvJ = QKV[j];
                        double dotVal = 0.0;
                        for (int k = 0; k < headDim; k++) {
                            dotVal += qkvI[hOff + k] * qkvJ[768 + hOff + k];
                        }
                        scoresBuf[j] = dotVal * 0.125;
                    }
                    double maxScore = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < bound; j++) {
                        if (scoresBuf[j] > maxScore) maxScore = scoresBuf[j];
                    }
                    double expSum = 0.0;
                    for (int j = 0; j < bound; j++) {
                        expsBuf[j] = Math.exp(scoresBuf[j] - maxScore);
                        expSum += expsBuf[j];
                    }
                    for (int j = 0; j < bound; j++) {
                        double prob = expsBuf[j] / expSum;
                        float[] qkvPos = QKV[j];
                        int vOff = 1536 + hOff;
                        for (int k = 0; k < headDim; k++) {
                            AttnMerged[pos][hOff + k] += (float) (prob * qkvPos[vOff + k]);
                        }
                    }
                }
            }

            // 4. Attn Projection: AttnMerged @ cPrW [768 x 768] + cPrB
            float[][] AttnProj = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                float[] r = AttnMerged[pos];
                for (int j = 0; j < embDim; j++) projAcc[j] = cPrB[j];
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * embDim;
                    for (int j = 0; j < embDim; j++) {
                        projAcc[j] += v * cPrW[rowOff + j];
                    }
                }
                for (int j = 0; j < embDim; j++) {
                    AttnProj[pos][j] = (float) clamp(projAcc[j]);
                }
            }

            // 5. Residual 1
            float[][] MRes1 = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                for (int k = 0; k < embDim; k++) {
                    MRes1[pos][k] = (float) clamp(CurrM[pos][k] + AttnProj[pos][k]);
                }
            }

            // 6. LayerNorm 2
            float[][] MNorm2 = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                float[] r = MRes1[pos];
                double sum = 0.0;
                for (int i = 0; i < embDim; i++) sum += r[i];
                double mean = sum / embDim;
                double varSum = 0.0;
                for (int i = 0; i < embDim; i++) {
                    double d = r[i] - mean;
                    varSum += d * d;
                }
                double std = Math.sqrt((varSum / embDim) + 1e-5);
                for (int i = 0; i < embDim; i++) {
                    MNorm2[pos][i] = (float) clamp(((r[i] - mean) / std) * g2[i] + b2[i]);
                }
            }

            // 7. MLP Block: MNorm2 @ fcW [768 x 3072] + fcB -> GELU -> @ prW [3072 x 768] + prB
            float[][] MMlp = new float[S][embDim];
            float[] fcOut = new float[3072];
            for (int pos = 0; pos < S; pos++) {
                float[] r = MNorm2[pos];
                for (int j = 0; j < 3072; j++) fcAcc[j] = fcB[j];
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * 3072;
                    for (int j = 0; j < 3072; j++) {
                        fcAcc[j] += v * fcW[rowOff + j];
                    }
                }
                for (int j = 0; j < 3072; j++) {
                    double x = fcAcc[j];
                    double geluV = 0.5 * x * (1.0 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
                    fcOut[j] = (float) clamp(geluV);
                }

                for (int j = 0; j < embDim; j++) projAcc[j] = prB[j];
                for (int i = 0; i < 3072; i++) {
                    double v = fcOut[i];
                    int rowOff = i * embDim;
                    for (int j = 0; j < embDim; j++) {
                        projAcc[j] += v * prW[rowOff + j];
                    }
                }
                for (int j = 0; j < embDim; j++) {
                    MMlp[pos][j] = (float) clamp(projAcc[j]);
                }
            }

            // 8. Residual 2
            for (int pos = 0; pos < S; pos++) {
                for (int k = 0; k < embDim; k++) {
                    CurrM[pos][k] = (float) clamp(MRes1[pos][k] + MMlp[pos][k]);
                }
            }
        }

        // Final LayerNorm on last position S-1
        float[] lastRow = CurrM[S - 1];
        float[] finalNormed = new float[embDim];
        double sum = 0.0;
        for (int i = 0; i < embDim; i++) sum += lastRow[i];
        double mean = sum / embDim;
        double varSum = 0.0;
        for (int i = 0; i < embDim; i++) {
            double d = lastRow[i] - mean;
            varSum += d * d;
        }
        double std = Math.sqrt((varSum / embDim) + 1e-5);
        for (int i = 0; i < embDim; i++) {
            finalNormed[i] = (float) clamp(((lastRow[i] - mean) / std) * lnFG[i] + lnFB[i]);
        }

        return finalNormed;
    }
}
