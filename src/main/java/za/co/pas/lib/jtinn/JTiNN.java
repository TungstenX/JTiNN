/*
 * Java implementation of Gustav Louw's The tiny neural network library (Tinn),
 * written in C
 * For more information: https://github.com/glouw/tinn
 */
package za.co.pas.lib.jtinn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

/**
 * Straight as possible conversion of Tinn from https://github.com/glouw/tinn
 *
 * @author Andr&eacute; Labuschagn&eacute; <andre@ParanoidAndroid.co.za>
 */
public class JTiNN {

    public static Logger LOG;

    static {
        InputStream stream = JTiNN.class.getClassLoader().
                getResourceAsStream("logging.properties");
        try {
            LogManager.getLogManager().readConfiguration(stream);
            LOG = Logger.getLogger(JTiNN.class.getName());
        } catch (IOException e) {
            System.err.println("Error while setting up logger: " + e.toString());
        }
    }
    private static final Random RANDOM = new Random(System.currentTimeMillis());

    float[] w; // All the weights.
    float[] x; // Hidden to output layer weights.
    float[] b; // Biases.
    float[] h; // Hidden layer.
    float[] o; // Output layer.

    int nb; // Number of biases - always two - Tinn only supports a single hidden layer.
    int nw; // Number of weights.

    int nips; // Number of inputs.
    int nhid; // Number of hidden neurons.
    int nops; // Number of outputs.

    public static float err(final float a, final float b) {
        return 0.5f * (float) Math.pow(a - b, 2.0f);
    }

    public static float pderr(final float a, final float b) {
        return a - b;
    }

    public static float toterr(final float[] tg, final float[] o, final int size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += err(tg[i], o[i]);
        }
        return sum;
    }

    public static float act(final float a) {
        return 1.0f / (1.0f + (float) Math.exp(a));
    }

    public static float pdact(final float a) {
        return a * (1.0f - a);
    }

    public static float frand() {
        return RANDOM.nextFloat();
    }

    public static void bprop(final JTiNN t, final float[] in, final float[] tg, float rate) {
        for (int i = 0; i < t.nhid; i++) {
            float sum = 0.0f;
            // Calculate total error change with respect to output.
            for (int j = 0; j < t.nops; j++) {
                final float _a = pderr(t.o[j], tg[j]);
                final float _b = pdact(t.o[j]);
                sum += _a * _b * t.x[j * t.nhid + i];
                // Correct weights in hidden to output layer.
                t.x[j * t.nhid + i] -= rate * _a * _b * t.h[i];
            }
            // Correct weights in input to hidden layer.
            for (int j = 0; j < t.nips; j++) {
                t.w[i * t.nips + j] -= rate * sum * pdact(t.h[i]) * in[j];
            }
        }
    }

    public static void fprop(final JTiNN t, final float[] in) {
        // Calculate hidden layer neuron values.
        for (int i = 0; i < t.nhid; i++) {
            float sum = 0.0f;
            for (int j = 0; j < t.nips; j++) {
                sum += in[j] * t.w[i * t.nips + j];
            }
            t.h[i] = act(sum + t.b[0]);
        }
        // Calculate output layer neuron values.
        for (int i = 0; i < t.nops; i++) {
            float sum = 0.0f;
            for (int j = 0; j < t.nhid; j++) {
                sum += t.h[j] * t.x[i * t.nhid + j];
            }
            t.o[i] = act(sum + t.b[1]);
        }
    }

    public static void twrand(final JTiNN t) {
        for (int i = 0; i < t.nw; i++) {
            t.w[i] = frand() - 0.5f;
        }
        for (int i = 0; i < t.nb; i++) {
            t.b[i] = frand() - 0.5f;
        }
    }

    public void bomb(String message, String... parts) {
        String s = message;
        if (parts != null && parts.length > 0) {
            s = String.format(s, (Object[]) parts);
        }
        LOG.log(Level.SEVERE, s);
        System.exit(-1);
    }

    public static float[] xtpredict(final JTiNN t, final float[] in) {
        fprop(t, in);
        return t.o;
    }

    public static float xttrain(final JTiNN t, final float[] in, final float[] tg, float rate) {
        fprop(t, in);
        bprop(t, in, tg, rate);
        return toterr(tg, t.o, t.nops);
    }

    public static JTiNN xtbuild(final int nips, final int nhid, final int nops) {
        JTiNN t = new JTiNN();
        // Tinn only supports one hidden layer so there are two biases.
        t.nb = 2;
        t.nw = nhid * (nips + nops);
        t.w = new float[t.nw];
        t.x = new float[t.w.length + nhid * nips];
        t.b = new float[t.nb];
        t.h = new float[nhid];
        t.o = new float[nops];
        t.nips = nips;
        t.nhid = nhid;
        t.nops = nops;
        twrand(t);
        return t;
    }

    public static void xtsave(final JTiNN t, final String path) {
        File file = new File(path);
        try (FileOutputStream fos = new FileOutputStream(file)) {
            // Header.
            String s = String.format("%d %d %d\n", t.nips, t.nhid, t.nops);
            fos.write(s.getBytes());
            // Biases and weights.
            for (int i = 0; i < t.nb; i++) {
                s = Float.toString(t.b[i]) + "\n";
                fos.write(s.getBytes());
            }
            for (int i = 0; i < t.nw; i++) {
                s = Float.toString(t.w[i]) + "\n";
                fos.write(s.getBytes());
            }
        } catch (IOException ex) {
            LOG.log(Level.SEVERE, "Error while saving file: {0}", ex.toString());
        }
    }

    public static JTiNN xtload(String path) {
        int _nips;
        int _nhid;
        int _nops;
        File file = new File(path);
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            //header
            String readLine = reader.readLine();
            String[] headerParts = readLine.split(" ");
            _nips = Integer.parseInt(headerParts[0]);
            _nhid = Integer.parseInt(headerParts[1]);
            _nops = Integer.parseInt(headerParts[2]);
            // A new tinn is returned.
            final JTiNN t = xtbuild(_nips, _nhid, _nops);
            // Biases and weights.
            for (int i = 0; i < t.nb; i++) {
                readLine = reader.readLine();
                t.b[i] = Float.parseFloat(readLine);
            }
            for (int i = 0; i < t.nw; i++) {
                readLine = reader.readLine();
                t.w[i] = Float.parseFloat(readLine);
            }
            return t;
        } catch (IOException ex) {
            LOG.log(Level.SEVERE, "Error while loading file: {0}", ex.toString());
        }
        return null;
    }
}
