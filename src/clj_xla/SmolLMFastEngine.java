package clj_xla;

public class SmolLMFastEngine {

    private static double clamp(double d) {
        if (Double.isNaN(d)) return 0.0;
        if (d > 3.4028234663852886E38) return 3.4028234663852886E38;
        if (d < -3.4028234663852886E38) return -3.4028234663852886E38;
        return d;
    }

    public static float[] evalSequence(float[][] X,
                                      float[][] inputLnW,
                                      float[][] qW, float[][] kW, float[][] vW, float[][] oW,
                                      float[][] postAttnLnW,
                                      float[][] gateW, float[][] upW, float[][] downW,
                                      float[] finalNormW) {
        int S = X.length;
        int embDim = 576;
        int numHeads = 9;
        int numKvHeads = 3;
        int headDim = 64;
        int intermediateSize = 1536;
        int numLayers = inputLnW.length;

        float[][] CurrM = new float[S][embDim];
        for (int s = 0; s < S; s++) {
            System.arraycopy(X[s], 0, CurrM[s], 0, embDim);
        }

        double[] qAcc = new double[embDim];
        double[] kAcc = new double[192];
        double[] vAcc = new double[192];
        double[] oAcc = new double[embDim];
        double[] gateAcc = new double[intermediateSize];
        double[] upAcc = new double[intermediateSize];
        double[] downAcc = new double[embDim];

        double[] scoresBuf = new double[2048];
        double[] expsBuf = new double[2048];

        // Precompute RoPE cos/sin frequencies for max seq len
        double[][] cosTable = new double[S][32];
        double[][] sinTable = new double[S][32];
        for (int pos = 0; pos < S; pos++) {
            for (int i = 0; i < 32; i++) {
                double freq = 1.0 / Math.pow(10000.0, (2.0 * i) / 64.0);
                double val = pos * freq;
                cosTable[pos][i] = Math.cos(val);
                sinTable[pos][i] = Math.sin(val);
            }
        }

        for (int l = 0; l < numLayers; l++) {
            float[] inLn = inputLnW[l];
            float[] qw = qW[l], kw = kW[l], vw = vW[l], ow = oW[l];
            float[] postLn = postAttnLnW[l];
            float[] gw = gateW[l], uw = upW[l], dw = downW[l];

            // 1. Input RMSNorm
            float[][] MNorm1 = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                float[] r = CurrM[pos];
                double sumSq = 0.0;
                for (int i = 0; i < embDim; i++) sumSq += r[i] * r[i];
                double rms = Math.sqrt((sumSq / embDim) + 1e-5);
                for (int i = 0; i < embDim; i++) {
                    MNorm1[pos][i] = (float) clamp((r[i] / rms) * inLn[i]);
                }
            }

            // 2. Q, K, V Projections
            float[][] Q = new float[S][embDim];
            float[][] K = new float[S][192];
            float[][] V = new float[S][192];

            for (int pos = 0; pos < S; pos++) {
                float[] r = MNorm1[pos];

                // Q Projection: 576 -> 576
                for (int j = 0; j < embDim; j++) qAcc[j] = 0.0;
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * embDim;
                    for (int j = 0; j < embDim; j++) {
                        qAcc[j] += v * qw[rowOff + j];
                    }
                }
                for (int j = 0; j < embDim; j++) Q[pos][j] = (float) clamp(qAcc[j]);

                // K Projection: 576 -> 192
                for (int j = 0; j < 192; j++) kAcc[j] = 0.0;
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * 192;
                    for (int j = 0; j < 192; j++) {
                        kAcc[j] += v * kw[rowOff + j];
                    }
                }
                for (int j = 0; j < 192; j++) K[pos][j] = (float) clamp(kAcc[j]);

                // V Projection: 576 -> 192
                for (int j = 0; j < 192; j++) vAcc[j] = 0.0;
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * 192;
                    for (int j = 0; j < 192; j++) {
                        vAcc[j] += v * vw[rowOff + j];
                    }
                }
                for (int j = 0; j < 192; j++) V[pos][j] = (float) clamp(vAcc[j]);
            }

            // 3. Apply RoPE to Q and K
            for (int pos = 0; pos < S; pos++) {
                double[] cosP = cosTable[pos];
                double[] sinP = sinTable[pos];

                // Rotate Q (9 heads)
                for (int h = 0; h < numHeads; h++) {
                    int hOff = h * headDim;
                    for (int i = 0; i < 32; i++) {
                        double q0 = Q[pos][hOff + i];
                        double q1 = Q[pos][hOff + i + 32];
                        double c = cosP[i], s = sinP[i];
                        Q[pos][hOff + i] = (float) clamp(q0 * c - q1 * s);
                        Q[pos][hOff + i + 32] = (float) clamp(q0 * s + q1 * c);
                    }
                }

                // Rotate K (3 heads)
                for (int h = 0; h < numKvHeads; h++) {
                    int hOff = h * headDim;
                    for (int i = 0; i < 32; i++) {
                        double k0 = K[pos][hOff + i];
                        double k1 = K[pos][hOff + i + 32];
                        double c = cosP[i], s = sinP[i];
                        K[pos][hOff + i] = (float) clamp(k0 * c - k1 * s);
                        K[pos][hOff + i + 32] = (float) clamp(k0 * s + k1 * c);
                    }
                }
            }

            // 4. GQA Causal Self-Attention
            float[][] AttnMerged = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                for (int h = 0; h < numHeads; h++) {
                    int qHeadOff = h * headDim;
                    int kvHeadOff = (h / 3) * headDim;
                    int bound = pos + 1;

                    for (int j = 0; j < bound; j++) {
                        double dotVal = 0.0;
                        for (int k = 0; k < headDim; k++) {
                            dotVal += Q[pos][qHeadOff + k] * K[j][kvHeadOff + k];
                        }
                        scoresBuf[j] = dotVal * 0.125; // 1 / sqrt(64)
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
                        for (int k = 0; k < headDim; k++) {
                            AttnMerged[pos][qHeadOff + k] += (float) (prob * V[j][kvHeadOff + k]);
                        }
                    }
                }
            }

            // 5. Attn Output Projection: 576 -> 576
            float[][] AttnProj = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                float[] r = AttnMerged[pos];
                for (int j = 0; j < embDim; j++) oAcc[j] = 0.0;
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * embDim;
                    for (int j = 0; j < embDim; j++) {
                        oAcc[j] += v * ow[rowOff + j];
                    }
                }
                for (int j = 0; j < embDim; j++) AttnProj[pos][j] = (float) clamp(oAcc[j]);
            }

            // 6. Residual 1
            float[][] MRes1 = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                for (int k = 0; k < embDim; k++) {
                    MRes1[pos][k] = (float) clamp(CurrM[pos][k] + AttnProj[pos][k]);
                }
            }

            // 7. Post-Attention RMSNorm
            float[][] MNorm2 = new float[S][embDim];
            for (int pos = 0; pos < S; pos++) {
                float[] r = MRes1[pos];
                double sumSq = 0.0;
                for (int i = 0; i < embDim; i++) sumSq += r[i] * r[i];
                double rms = Math.sqrt((sumSq / embDim) + 1e-5);
                for (int i = 0; i < embDim; i++) {
                    MNorm2[pos][i] = (float) clamp((r[i] / rms) * postLn[i]);
                }
            }

            // 8. SwiGLU MLP Block
            float[][] MMlp = new float[S][embDim];
            float[] actUpOut = new float[intermediateSize];
            for (int pos = 0; pos < S; pos++) {
                float[] r = MNorm2[pos];

                // Gate & Up projections: 576 -> 1536
                for (int j = 0; j < intermediateSize; j++) {
                    gateAcc[j] = 0.0;
                    upAcc[j] = 0.0;
                }
                for (int i = 0; i < embDim; i++) {
                    double v = r[i];
                    int rowOff = i * intermediateSize;
                    for (int j = 0; j < intermediateSize; j++) {
                        gateAcc[j] += v * gw[rowOff + j];
                        upAcc[j] += v * uw[rowOff + j];
                    }
                }

                for (int j = 0; j < intermediateSize; j++) {
                    double gv = gateAcc[j];
                    double uv = upAcc[j];
                    double siluG = gv / (1.0 + Math.exp(-gv));
                    actUpOut[j] = (float) clamp(siluG * uv);
                }

                // Down projection: 1536 -> 576
                for (int j = 0; j < embDim; j++) downAcc[j] = 0.0;
                for (int i = 0; i < intermediateSize; i++) {
                    double v = actUpOut[i];
                    int rowOff = i * embDim;
                    for (int j = 0; j < embDim; j++) {
                        downAcc[j] += v * dw[rowOff + j];
                    }
                }
                for (int j = 0; j < embDim; j++) MMlp[pos][j] = (float) clamp(downAcc[j]);
            }

            // 9. Residual 2
            for (int pos = 0; pos < S; pos++) {
                for (int k = 0; k < embDim; k++) {
                    CurrM[pos][k] = (float) clamp(MRes1[pos][k] + MMlp[pos][k]);
                }
            }
        }

        // Final RMSNorm on last position S-1
        float[] lastRow = CurrM[S - 1];
        float[] finalNormed = new float[embDim];
        double sumSq = 0.0;
        for (int i = 0; i < embDim; i++) sumSq += lastRow[i] * lastRow[i];
        double rms = Math.sqrt((sumSq / embDim) + 1e-5);
        for (int i = 0; i < embDim; i++) {
            finalNormed[i] = (float) clamp((lastRow[i] / rms) * finalNormW[i]);
        }

        return finalNormed;
    }
}
