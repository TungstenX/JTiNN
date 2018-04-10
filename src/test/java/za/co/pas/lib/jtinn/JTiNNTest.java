package za.co.pas.lib.jtinn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.assertTrue;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Andr&eacute; Labuschagn&eacute; <andre@ParanoidAndroid.co.za>
 */
public class JTiNNTest {

    private static final String DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data";
    public static Logger LOG;
    static {
        InputStream stream = JTiNNTest.class.getClassLoader().
                getResourceAsStream("logging.properties");
        try {
            LogManager.getLogManager().readConfiguration(stream);
            LOG = Logger.getLogger(JTiNNTest.class.getName());
        } catch (IOException e) {
            System.err.println("Error while setting up logger: " + e.toString());
        }
    }
    private static final Random RANDOM = new Random(System.currentTimeMillis());

    public JTiNNTest() {
    }

    @BeforeClass
    public static void setUpClass() {
        URL url;
        try {
            // get URL content
            url = new URL(DATA_URL);
            URLConnection conn = url.openConnection();
            // open the stream and put it into BufferedReader
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream()))) {
                String inputLine;

                //save to this filename
                String fileName = "semeion.data";
                File file = new File(fileName);

                if (!file.exists()) {
                    file.createNewFile();
                }

                //use FileWriter to write file
                FileWriter fw = new FileWriter(file.getAbsoluteFile());
                try (BufferedWriter bw = new BufferedWriter(fw)) {
                    while ((inputLine = br.readLine()) != null) {
                        bw.write(inputLine);
                    }
                }
            }
            LOG.log(Level.INFO, "Done");
        } catch (MalformedURLException e) {
            LOG.log(Level.SEVERE, "Error in url: {0}", e.toString());
        } catch (IOException e) {
            LOG.log(Level.SEVERE, "IO Error: {0}", e.toString());
        }
    }

    @AfterClass
    public static void tearDownClass() {
        try {
            File file = new File("semeion.data");
            if (file.delete()) {
                LOG.log(Level.INFO, "{0} is deleted!", file.getName());
            } else {
                LOG.log(Level.WARNING, "Delete operation is failed for {0}", file.getName());
            }
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "Error while deleting file: {0}", e.toString());
        }
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * From Tinn's test
     */
    @Test
    public void testExample() throws IOException {
        // Input and output size is harded coded here as machine learning
        // repositories usually don't include the input and output size in the data itself.
        final int nips = 256;
        final int nops = 10;
        // Hyper Parameters.
        // Learning rate is annealed and thus not constant.
        // It can be fine tuned along with the number of hidden layers.
        // Feel free to modify the anneal rate as well.
        final int nhid = 28;
        float rate = 1.0f;
        final float anneal = 0.99f;
        // Load the training set.
        final Data data = build("semeion.data", nips, nops);
        // Train, baby, train.
        final JTiNN tinn = JTiNN.xtbuild(nips, nhid, nops);
        for (int i = 0; i < 100; i++) {
            shuffle(data);
            float error = 0.0f;
            for (int j = 0; j < data.rows; j++) {
                float[] in = data.in[j];
                float[] tg  = data.tg[j];
                error += JTiNN.xttrain(tinn, in, tg, rate);
            }
            LOG.log(Level.INFO,"error {0} :: learning rate {1}\n", new Object[]{String.format("%.12f", (double) error / data.rows), (double) rate});
            rate *= anneal;
        }
        // This is how you save the neural network to disk.
        JTiNN.xtsave(tinn, "saved.tinn");
        
        // This is how you load the neural network from disk.
        final JTiNN loaded = JTiNN.xtload("saved.tinn");
        // Now we do a prediction with the neural network we loaded from disk.
        // Ideally, we would also load a testing set to make the prediction with,
        // but for the sake of brevity here we just reuse the training set from earlier.
        float[] in  = data.in[0];
        float[] tg  = data.tg[0];
        float[] pd  = JTiNN.xtpredict(loaded, in);
        StringBuilder sb = new StringBuilder("\n");
        for (int i = 0; i < data.nops; i++) {
            sb.append(String.format("%.12f",tg[i])).append(" ");
        }
        sb.append("\n");
        for (int i = 0; i < data.nops; i++) {
            sb.append(String.format("%.12f",pd[i])).append(" ");
        }
        sb.append("\n");
        LOG.log(Level.INFO,sb.toString());
        // All done. Let's clean up.
        assertTrue(true);
    }

    private Data build(String path, final int nips, final int nops) throws FileNotFoundException, IOException {
        File file = new File(path);
        List<String> lines = new LinkedList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            //header
            lines.add(reader.readLine());
        }
        int rows = lines.size();
        Data data = ndata(nips, nops, rows);
        for (int row = 0; row < rows; row++) {
            parse(data, lines.get(row), row);
        }
        return data;
    }

    private float[][] new2d(final int rows, final int cols) {
        float[][] row = new float[rows][cols];
        return row;
    }

    private Data ndata(final int nips, final int nops, final int rows) {
        Data data = new Data(new2d(rows, nips), new2d(rows, nops), nips, nops, rows);
        return data;
    }

    private void shuffle(final Data d) {
        for (int a = 0; a < d.rows; a++) {
            final int b = RANDOM.nextInt() % d.rows;
            float[] ot = d.tg[a];
            float[] it = d.in[a];
            // Swap output.
            d.tg[a] = d.tg[b];
            d.tg[b] = ot;
            // Swap input.
            d.in[a] = d.in[b];
            d.in[b] = it;
        }
    }

    private void parse(final Data data, String line, final int row) {
        final int cols = data.nips + data.nops;
        String[] parts = line.split(" ");
        for (int col = 0; col < cols; col++) {
            final float val = Float.parseFloat(parts[col]);
            if (col < data.nips) {
                data.in[row][col] = val;
            } else {
                data.tg[row][col - data.nips] = val;
            }
        }
    }

    private class Data {

        float[][] in;
        float[][] tg;
        int nips;
        int nops;
        int rows;

        public Data(float[][] in, float[][] tg, int nips, int nops, int rows) {
            this.in = in;
            this.tg = tg;
            this.nips = nips;
            this.nops = nops;
            this.rows = rows;
        }
    }
}
