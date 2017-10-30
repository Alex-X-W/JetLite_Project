// Load from pretrained python Tensorflow event tagging model
// Author: Xuan Wang

package edu.nyu.jetlite;

import edu.nyu.jetlite.tipster.*;
import java.io.*;
import java.util.*;

import org.apache.commons.lang.ArrayUtils;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.util.stream.Collectors;
import java.util.Map.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 *  Identify ACE events.
 *  <p>
 *  This 'skeleton' tagger does not include finding event arguments.
 */

public class NeuralTagModel {
    private byte[] graphDef;
    private JSONObject mappings;
    private Map<Integer, String> idToChar, idToTag, idToWord;
    private Map<String, Integer> charToId, TagToId, wordToId;

    public NeuralTagModel(Properties config) throws IOException {
        String modelDir = config.getProperty("NeuralTagModel.modelDir");
        String modelFileName = config.getProperty("NeuralTagModel.modelFileName");
        String modelMappings = config.getProperty("NeuralTagModel.mappings");

        // read model mappings
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(modelMappings));
            JSONObject jsonObject = (JSONObject) obj;

            idToChar = (Map<Integer, String>) jsonObject.get("id_to_char");
            idToTag = (Map<Integer, String>) jsonObject.get("id_to_tag");
            idToWord = (Map<Integer, String>) jsonObject.get("id_to_word");

            charToId = invert(idToChar);
            TagToId = invert(idToTag);
            wordToId = invert(idToWord);

        } catch (Exception e) {
            System.out.println("Failed reading TF model config mappings...");
            e.printStackTrace();
        }

        graphDef = readAllBytesOrExit(Paths.get(modelDir, modelFileName));
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    public String[] predict(Document doc, Sentence sent) {
        Tensor input = createInput(doc, sent);
        int[] result = executeGraph(graphDef, input);
        String[] tags = convertToTags(result);

    }

    private static int[] executeGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
                return result.copyTo(new int[2][2]);
            }
        }
    }

    private static String[] convertToTags(int[] idxList) {

    }

    /*
    feed_dict_[model.word_ids]     = input_[0]
    feed_dict_[model.word_pos_ids] = input_[1]
    feed_dict_[model.char_for_ids] = input_[2]
    feed_dict_[model.char_rev_ids] = input_[3]
    feed_dict_[model.char_pos_ids] = input_[4]
    self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids') # shape:[batch_size, max_word_len]
    self.word_pos_ids = tf.placeholder(tf.int32, shape=[None], name='word_pos_ids') # shape: [batch_size]
    self.char_for_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_for_ids') # shape: [batch_size, word_max_len, char_max_len]
    self.char_rev_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_rev_ids') # shape: [batch_size, word_max_len, char_max_len]
    self.char_pos_ids = tf.placeholder(tf.int32, shape=[None, None], name='char_pos_ids') # shape: [batch_size*word_max_len, char_max_len]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    */
    private Tensor createInput(Document doc, Sentence sent) {
        Span span = sent.span();
        int posn = span.start();

        int[][] wordIds
        int[] wordPosIds;
        int[][][] charForIds;
        int[][][] charRevIds;
        int[][] charPosIds;

        // retrieve tokens from sentence span
        ArrayList<String> tokens = new ArrayList<>();
        ArrayList<Integer> tokenIds = new ArrayList<>();
        int charMaxLen = 0;
        while (posn < span.end()) {
            Annotation tokenAnnotation = doc.tokenAt(posn);
            if (tokenAnnotation == null)
                break;
            String tokenText = doc.normalizedText(tokenAnnotation).trim();
            tokens.add(tokenText);
            if (tokenText.length() > charMaxLen) {
                charMaxLen = tokenText.length();
            }
            Integer tokenId = wordToId.get(tokenText);
            tokenIds.add(tokenId == null ? wordToId.get("<UNK>"): tokenId);
            posn = tokenAnnotation.end();
        }
        wordIds = new int[1][tokenIds.size()];
        wordIds[0] = ArrayUtils.toPrimitive(tokenIds.toArray(new Integer[0]));
        wordPosIds = new int[1];
        wordPosIds[0] = tokenIds.size() - 1;

        charForIds = new int[1][tokenIds.size()][charMaxLen];
        charRevIds = new int[1][tokenIds.size()][charMaxLen];
        charPosIds = new int[tokenIds.size()][2];

        int[] charIds = new int[charMaxLen];
        int cnt = 0;
        for (String e: tokens) {
            int pos = e.length() - 1;

        }


        // word_ids -> Tensor
        try (Graph g = new Graph()) {
            tf_intergration.GraphBuilder b = new tf_intergration.GraphBuilder(g);
            // Since the graph is being constructed once per execution here, we can use a constant for the
            // input image. If the graph were to be re-used for multiple input images, a placeholder would
            // have been more appropriate.
            final Output output = b.constant("input", input);
            try (Session s = new Session(g)) {
                Tensor inputWordIds = s.runner().fetch(output.op().name()).run().get(0);
            }
        }

    }

    public static <V, K> Map<V, K> invert(Map<K, V> map) {
        return map.entrySet()
                .stream()
                .collect(Collectors.toMap(Entry::getValue, c -> c.getKey()));
    }

}
