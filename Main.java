import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.util.MathArrays;
import ws.schild.jave.Encoder;
import ws.schild.jave.EncoderException;
import ws.schild.jave.MultimediaObject;
import ws.schild.jave.encode.AudioAttributes;
import ws.schild.jave.encode.EncodingAttributes;

import javax.imageio.ImageIO;
import javax.sound.sampled.*;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.logging.Level;
import java.util.logging.Logger;


enum DataType {
    TEXT,
    IMAGE,
    NUMERICAL,
    AUDIO,
    SENSOR
}

interface Preprocessor<T> extends Serializable {
    T preprocess(T data);
}

class TextPreprocessor implements Preprocessor<String> {
    @Override
    public String preprocess(String text) {
        text = text.toLowerCase().replaceAll("[^a-z ]", "");
        text = removeStopWords(text);
        return text;
    }

    private String removeStopWords(String text) {
        // Basic stop words removal logic
        List<String> stopWords = Arrays.asList("the", "and", "is", "of", "it", "in", "to", "a", "for");
        for (String stopWord : stopWords) {
            text = text.replaceAll("\\b" + stopWord + "\\b", "");
        }
        return text;
    }
}

class ImagePreprocessor implements Preprocessor<BufferedImage> {
    @Override
    public BufferedImage preprocess(BufferedImage image) {
        return applyGrayscaleConversion(image);
    }

    private BufferedImage applyGrayscaleConversion(BufferedImage image) {
        // Implement grayscale conversion logic
        BufferedImage resultImage = new BufferedImage(
                image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

        Graphics g = resultImage.getGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();

        return resultImage;
    }
}

class NumericalPreprocessor implements Preprocessor<double[]> {
    @Override
    public double[] preprocess(double[] numericalData) {
        double[] dataWithMissingValuesHandled = handleMissingValues(numericalData);
        normalize(dataWithMissingValuesHandled);
        return dataWithMissingValuesHandled; // Return the processed data
    }

    private void normalize(double[] numericalData) {
        // normalization logic here
        double mean = calculateMean(numericalData);
        double variance = calculateVariance(numericalData, mean);

        for (int i = 0; i < numericalData.length; i++) {
            numericalData[i] = (numericalData[i] - mean) / Math.sqrt(variance);
        }

    }

    private double[] handleMissingValues(double[] numericalData) {
        // Simple strategy: Replace missing values with the mean
        double mean = calculateMean(numericalData);
        for (int i = 0; i < numericalData.length; i++) {
            if (Double.isNaN(numericalData[i])) {
                numericalData[i] = mean;
            }
        }
        return numericalData;
    }

    private double calculateMean(double[] numericalData) {
        double sum = 0.0;
        int count = 0;
        for (double value : numericalData) {
            if (!Double.isNaN(value)) {
                sum += value;
                count++;
            }
        }
        return sum / count;
    }

    private double calculateVariance(double[] numericalData, double mean) {
        double sumSquaredDifferences = 0.0;
        int count = 0;
        for (double value : numericalData) {
            if (!Double.isNaN(value)) {
                sumSquaredDifferences += Math.pow(value - mean, 2);
                count++;
            }
        }
        return sumSquaredDifferences / count;
    }
}

class AudioPreprocessor implements Preprocessor<byte[]> {
    private static final double SPECTRAL_SUBTRACTION_FACTOR = 1.5;
    private static final Logger logger = Logger.getLogger(AudioPreprocessor.class.getName());

    @Override
    public byte[] preprocess(byte[] audioData) {
        File audioFile = chooseAudioFile();

        if (audioFile == null) {
            logger.warning("Audio file selection canceled or failed.");
            return audioData;
        }

        return readAndPreprocessAudioFile(audioFile);
    }


    private File chooseAudioFile() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Choose an audio file");
        fileChooser.setFileFilter(new FileNameExtensionFilter("Audio files", "wav", "mp3"));

        int userSelection = fileChooser.showOpenDialog(null);
        if (userSelection == JFileChooser.APPROVE_OPTION) {
            return fileChooser.getSelectedFile();
        } else {
            return null;
        }
    }

    private byte[] readAndPreprocessAudioFile(File audioFile) {
        try {
            String fileName = audioFile.getName().toLowerCase();
            if (fileName.endsWith(".wav")) {
                return preprocessWavFile(audioFile);
            } else if (fileName.endsWith(".mp3")) {
                return preprocessMp3File(audioFile);
            } else {
                logger.log(Level.WARNING, "Unsupported audio file format: {0}", fileName);
                return null;
            }
        } catch (IOException | EncoderException | UnsupportedAudioFileException | LineUnavailableException e) {
            logger.log(Level.SEVERE, "Error processing audio file: " + audioFile.getName(), e);
            e.printStackTrace();
            return null;
        }
    }


    private byte[] preprocessWavFile(File audioFile) throws IOException, UnsupportedAudioFileException {
        return getBytes(audioFile);
    }

    private byte[] getBytes(File audioFile) throws IOException, UnsupportedAudioFileException {
        try (AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(audioFile)) {
            int originalSize = (int) audioInputStream.getFrameLength();

            // Calculate the nearest power of 2 for padding
            int paddedSize = 1;
            while (paddedSize < originalSize) {
                paddedSize <<= 1;
            }

            // Pad the input array with zeros to the nearest power of 2
            byte[] buffer = new byte[paddedSize * audioInputStream.getFormat().getFrameSize()];

            int bytesRead = audioInputStream.read(buffer, 0, originalSize * audioInputStream.getFormat().getFrameSize());
            if (bytesRead == -1) {
                throw new IOException("End of stream reached while reading audio data.");
            }

            // Convert byte array to double array
            double[] audioSamples = byteArrayToDoubleArray(buffer);

            // Apply spectral subtraction for noise reduction
            double[] processedSamples = spectralSubtraction(audioSamples);

            // Convert back to byte array
            return doubleArrayToByteArray(processedSamples);
        }
    }


    private byte[] preprocessMp3File(File audioFile) throws IOException, EncoderException, UnsupportedAudioFileException, LineUnavailableException {
        File outputWavFile = File.createTempFile("output", ".wav");
        try {
            convertMp3ToWav(audioFile, outputWavFile);

            return applySpectralSubtraction(outputWavFile);
        } finally {
            // Delete the temporary WAV file, and handle the result
            if (!outputWavFile.delete()) {
                System.err.println("Failed to delete the temporary WAV file: " + outputWavFile.getAbsolutePath());
            }
        }
    }

    private void convertMp3ToWav(File mp3File, File wavOutputFile) throws EncoderException {
        AudioAttributes audio = new AudioAttributes();
        audio.setCodec("pcm_s16le");
        audio.setBitRate(256000);
        audio.setChannels(2);
        audio.setSamplingRate(44100);

        EncodingAttributes attrs = new EncodingAttributes();
        attrs.setOutputFormat("wav");
        attrs.setAudioAttributes(audio);

        Encoder encoder = new Encoder();
        encoder.encode(new MultimediaObject(mp3File), wavOutputFile, attrs);
    }

    private byte[] applySpectralSubtraction(File wavFile) throws IOException, UnsupportedAudioFileException {
        return getBytes(wavFile);
    }

    private double[] byteArrayToDoubleArray(byte[] byteArray) {
        ByteBuffer byteBuffer = ByteBuffer.wrap(byteArray);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);

        int numSamples = byteArray.length / 2; // Assuming 16-bit audio
        double[] result = new double[numSamples];

        for (int i = 0; i < numSamples; i++) {
            result[i] = byteBuffer.getShort() / 32768.0;
        }

        return result;
    }

    private byte[] doubleArrayToByteArray(double[] doubleArray) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(doubleArray.length * 2); // 16-bit audio
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);

        for (double value : doubleArray) {
            short shortValue = (short) (value * 32767.0);
            byteBuffer.putShort(shortValue);
        }

        return byteBuffer.array();
    }


public static int findNearestPowerOf2(int n) {
    if (n <= 0) {
        throw new IllegalArgumentException("Input must be a positive integer");
    }

    int power = 1;

    while (power < n) {
        power <<= 1;
    }

    int nextPower = power;
    int previousPower = power >> 1;

    // Determine the nearest power of 2
    if (n - previousPower <= nextPower - n) {
        return previousPower;
    } else {
        return nextPower;
    }
}
    private double[] spectralSubtraction(double[] audioSamples) {
        // Find the nearest power of 2 for padding
        int inputNumber = ((audioSamples.length-2)/10);

        int nearestPowerOf2 = findNearestPowerOf2(inputNumber);

        // Pad the input array with zeros to the nearest power of 2
        double[] paddedSamples = Arrays.copyOf(audioSamples, nearestPowerOf2);

        // Apply Fast Fourier Transform (FFT) to get the spectrum
        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] spectrum = transformer.transform(paddedSamples, org.apache.commons.math3.transform.TransformType.FORWARD);

        // Estimate noise spectrum from a portion of the audio (e.g., 10%)
        int noiseEstimationSize = paddedSamples.length;
        Complex[] noiseSpectrum = transformer.transform(Arrays.copyOfRange(paddedSamples, 0, noiseEstimationSize), org.apache.commons.math3.transform.TransformType.FORWARD);

        // Perform spectral subtraction
        for (int i = 0; i < spectrum.length; i++) {
            // Reduce the magnitude of the spectrum by a factor, but ensure it doesn't go below zero
            double magnitude = Math.max(spectrum[i].getReal() - SPECTRAL_SUBTRACTION_FACTOR * noiseSpectrum[i].getReal(), 0);
            spectrum[i] = new Complex(magnitude, 0);
        }

        // Apply Inverse FFT to get the time-domain signal
        Complex[] processedSpectrum = transformer.transform(spectrum, org.apache.commons.math3.transform.TransformType.INVERSE);
        double[] processedSamples = new double[processedSpectrum.length];
        for (int i = 0; i < processedSpectrum.length; i++) {
            processedSamples[i] = processedSpectrum[i].getReal();
        }

        // Return the processed samples, considering the original length
        return Arrays.copyOf(processedSamples, audioSamples.length);
    }

}

class SensorDataProcessor implements Preprocessor<double[]> {

    @Override
    public double[] preprocess(double[] sensorData) {
        filterOutliers(sensorData);
        sensorData = normalize(sensorData);
        return sensorData;
    }

    private void filterOutliers(double[] sensorData) {
        // Use DescriptiveStatistics to calculate mean and standard deviation
        DescriptiveStatistics stats = new DescriptiveStatistics(sensorData);

        double mean = stats.getMean();
        double stdDev = stats.getStandardDeviation();

        // Define a threshold (e.g., 3 times the standard deviation)
        double threshold = 3.0 * stdDev;

        // Filter out values beyond the threshold
        for (int i = 0; i < sensorData.length; i++) {
            if (Math.abs(sensorData[i] - mean) > threshold) {
                sensorData[i] = mean; // Replace outlier with mean
            }
        }

    }

    private double[] normalize(double[] sensorData) {
        // Use MathArrays.normalizeArray to normalize the data
        return MathArrays.normalizeArray(sensorData, 1.0);
    }
}
class HardwareAccess {

    public static double[] readSensorData() {
        // Perform sensor data reading
        // dummy array.
        return new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    public static double[] performCPUComputation() {
        // Perform CPU-based computation
        double[] inputData = readSensorData();

        // Normalize the array using parallel processing
        try (ForkJoinPool forkJoinPool = new ForkJoinPool()) {
            return forkJoinPool.invoke(new NormalizeTask(inputData));
        }
    }


    private static class NormalizeTask extends RecursiveTask<double[]> {
        private static final int THRESHOLD = 5; // Adjust the threshold as needed
        private final double[] data;

        public NormalizeTask(double[] data) {
            this.data = data;
        }

        @Override
        protected double[] compute() {
            if (data.length <= THRESHOLD) {
                // Perform normalization directly
                return MathArrays.normalizeArray(data, 1.0);
            } else {
                // Split the task into subtasks
                int mid = data.length / 2;
                NormalizeTask leftTask = new NormalizeTask(Arrays.copyOfRange(data, 0, mid));
                NormalizeTask rightTask = new NormalizeTask(Arrays.copyOfRange(data, mid, data.length));

                // Fork and join
                invokeAll(leftTask, rightTask);

                // Merge results
                double[] leftResult = leftTask.join();
                double[] rightResult = rightTask.join();

                // Merge the sub-results
                double[] mergedResult = new double[data.length];
                System.arraycopy(leftResult, 0, mergedResult, 0, leftResult.length);
                System.arraycopy(rightResult, 0, mergedResult, leftResult.length, rightResult.length);

                return mergedResult;
            }
        }
    }

}

class DataStorage<T> {
    private final List<T> data;

    public DataStorage() {
        this.data = new ArrayList<>();
    }

    public void addData(T newData) {
        data.add(newData);
    }

    public T getData(int index) {
        if (index < 0 || index >= data.size()) {
            // Handle the case when the index is out of bounds
            System.err.println("Index is out of bounds: " + index);
            return null;
        }
        return data.get(index);
    }


    public int getSize() {
        return data.size();
    }

    public void getDataSubset(double splitRatio, boolean isTraining) {
        int dataSize = data.size();
        int endIndex = (int) (splitRatio * dataSize);
        if (isTraining) {
            data.subList(0, endIndex).clear(); // Clear the sublist from the original data
        } else {
            data.subList(endIndex, dataSize).clear(); // Clear the sublist from the original data
        }
    }


    public List<List<T>> getBatches(int batchSize) {
        List<List<T>> batches = new ArrayList<>();
        for (int i = 0; i < data.size(); i += batchSize) {
            int endIndex = Math.min(i + batchSize, data.size());
            batches.add(new ArrayList<>(data.subList(i, endIndex)));
        }
        return batches;
    }
}

class DataPreprocessingLibrary {
    private static final Logger LOGGER = Logger.getLogger(DataPreprocessingLibrary.class.getName());
    private final DataStorage<String> textDataStorage;
    private final DataStorage<BufferedImage> imageDataStorage;
    private final DataStorage<double[]> numericalDataStorage;
    private final DataStorage<byte[]> audioDataStorage;
    private final DataStorage<double[]> sensorDataStorage;

    public DataPreprocessingLibrary() {
        this.textDataStorage = new DataStorage<>();
        this.imageDataStorage = new DataStorage<>();
        this.numericalDataStorage = new DataStorage<>();
        this.audioDataStorage = new DataStorage<>();
        this.sensorDataStorage = new DataStorage<>();
    }

    public <T> void addData(T data, DataType dataType) {
        Preprocessor<T> preprocessor = getPreprocessor(dataType);
        T preprocessedData = preprocessor.preprocess(data);
        storeData(preprocessedData, dataType);
    }

    public Object getData(int index, DataType dataType) {
        return switch (dataType) {
            case TEXT -> textDataStorage.getData(index);
            case IMAGE -> imageDataStorage.getData(index);
            case NUMERICAL -> numericalDataStorage.getData(index);
            case AUDIO -> audioDataStorage.getData(index);
            case SENSOR -> sensorDataStorage.getData(index);
        };
    }
    @SuppressWarnings("unchecked")
    private <T> Preprocessor<T> getPreprocessor(DataType dataType) {
        return switch (dataType) {
            case TEXT -> (Preprocessor<T>) new TextPreprocessor();
            case IMAGE -> (Preprocessor<T>) new ImagePreprocessor();
            case NUMERICAL -> (Preprocessor<T>) new NumericalPreprocessor();
            case AUDIO -> (Preprocessor<T>) new AudioPreprocessor();
            case SENSOR -> (Preprocessor<T>) new SensorDataProcessor();
        };
    }


    private void storeData(Object preprocessedData, DataType dataType) {
        switch (dataType) {
            case TEXT -> textDataStorage.addData((String) preprocessedData);
            case IMAGE -> imageDataStorage.addData((BufferedImage) preprocessedData);
            case NUMERICAL -> numericalDataStorage.addData((double[]) preprocessedData);
            case AUDIO -> audioDataStorage.addData((byte[]) preprocessedData);
            case SENSOR -> sensorDataStorage.addData((double[]) preprocessedData);
            default -> throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    public void readAndAddSensorData() {
        double[] sensorData = HardwareAccess.readSensorData();
        addData(sensorData, DataType.SENSOR);
    }

    public void splitData(double splitRatio) {
        textDataStorage.getDataSubset(splitRatio, true);
        textDataStorage.getDataSubset(splitRatio, false);

        imageDataStorage.getDataSubset(splitRatio, true);
        imageDataStorage.getDataSubset(splitRatio, false);

        numericalDataStorage.getDataSubset(splitRatio, true);
        numericalDataStorage.getDataSubset(splitRatio, false);

        audioDataStorage.getDataSubset(splitRatio, true);
        audioDataStorage.getDataSubset(splitRatio, false);

        sensorDataStorage.getDataSubset(splitRatio, true);
        sensorDataStorage.getDataSubset(splitRatio, false);

    }

    public List<List<String>> getTextTrainingBatches(int batchSize) {
        return textDataStorage.getBatches(batchSize);
    }

    public List<List<BufferedImage>> getImageTrainingBatches(int batchSize) {
        return imageDataStorage.getBatches(batchSize);
    }

    public List<List<double[]>> getNumericalTrainingBatches(int batchSize) {
        return numericalDataStorage.getBatches(batchSize);
    }

    public List<List<byte[]>> getAudioTrainingBatches(int batchSize) {
        return audioDataStorage.getBatches(batchSize);
    }

    public List<List<double[]>> getSensorTrainingBatches(int batchSize) {
        return sensorDataStorage.getBatches(batchSize);

    }

    private BufferedImage chooseImageFile() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Choose an image file");
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Image files", "jpg", "jpeg", "png", "gif");
        fileChooser.setFileFilter(filter);

        int userSelection = fileChooser.showOpenDialog(null);
        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            try {
                return ImageIO.read(selectedFile);
            } catch (IOException e) {
                LOGGER.log(Level.SEVERE, "Error reading image file", e);
                return null;
            }
        } else {
            return null;
        }
    }
    private void displayImage(BufferedImage image) {
        JFrame frame = new JFrame("Processed Image");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Resize the image to fit within a 500x500 box
        int targetWidth = 500;
        int targetHeight = 500;
        BufferedImage resizedImage = resizeImage(image, targetWidth, targetHeight);

        // Create a JLabel to display the resized image
        JLabel label = new JLabel(new ImageIcon(resizedImage));

        // Create a JScrollPane to handle scrolling if the image is too large
        JScrollPane scrollPane = new JScrollPane(label);

        // Set the preferred size of the scroll pane to achieve a medium-sized display
        scrollPane.setPreferredSize(new Dimension(targetWidth, targetHeight));

        frame.getContentPane().add(scrollPane);
        frame.pack();
        frame.setLocationRelativeTo(null); // Center the frame on the screen
        frame.setVisible(true);
    }

    private BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
        return resizedImage;
    }

    public void addImageFromFile() {
        BufferedImage selectedImage = chooseImageFile();
        if (selectedImage != null) {
            addData(selectedImage, DataType.IMAGE);
            int lastIndex = imageDataStorage.getSize()- 1;
            displayImage(imageDataStorage.getData(lastIndex));
        }
    }
    private static void playAudio(byte[] audioData) {
        try {
            // Set up an audio stream from the byte array
            AudioFormat audioFormat = new AudioFormat(44100, 16, 1, true, false);
            ByteArrayInputStream bais = new ByteArrayInputStream(audioData);
            AudioInputStream audioInputStream = new AudioInputStream(bais, audioFormat, audioData.length / audioFormat.getFrameSize());

            // Play the audio
            DataLine.Info info = new DataLine.Info(SourceDataLine.class, audioFormat);
            SourceDataLine line = (SourceDataLine) AudioSystem.getLine(info);
            line.open(audioFormat);
            line.start();

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = audioInputStream.read(buffer, 0, buffer.length)) != -1) {
                line.write(buffer, 0, bytesRead);
            }

            line.drain();
            line.close();
            audioInputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            DataPreprocessingLibrary library = new DataPreprocessingLibrary();
            String userInputText = getUserTextInput();
            library.addData(userInputText, DataType.TEXT);
            library.addImageFromFile();
            double[] numericalData = {1.0, 2.0, Double.NaN, 4.0, 5.0};
            library.addData(numericalData, DataType.NUMERICAL);
            byte[] simulatedAudio = {0, 1, 2, 3, 4};
            library.addData(simulatedAudio, DataType.AUDIO);
            library.readAndAddSensorData();
            System.out.println("Text Data: " + library.getData(0, DataType.TEXT));
            System.out.println("Image Data: " + library.getData(0, DataType.IMAGE));
            System.out.println("Numerical Data: " + arrayToString((double[]) library.getData(0, DataType.NUMERICAL)));

            byte[] audioData = (byte[]) library.getData(0, DataType.AUDIO);
            if (audioData != null && audioData.length > 0) {
                System.out.println("Audio processed successfuly, listen now playing.");
                playAudio(audioData);
            } else {
                System.out.println("Audio Data is null or empty.");
            }

            System.out.println("Sensor Data: " + arrayToString((double[]) library.getData(0, DataType.SENSOR)));
            double[] result = HardwareAccess.performCPUComputation();
            System.out.println("Result from CPU computation: " + Arrays.toString(result));
            library.splitData(0.8);
            // Get batches for each data type
            List<List<String>> textBatches = library.getTextTrainingBatches(10);
            List<List<BufferedImage>> imageBatches = library.getImageTrainingBatches(10);
            List<List<double[]>> numericalBatches = library.getNumericalTrainingBatches(10);
            List<List<byte[]>> audioBatches = library.getAudioTrainingBatches(10);
            List<List<double[]>> sensorBatches = library.getSensorTrainingBatches(10);

            // Print or process the batches as needed
            System.out.println("Text Batches: " + textBatches);
            System.out.println("Image Batches: " + imageBatches);
            System.out.println("Numerical Batches: " + numericalBatches);
            System.out.println("Audio Batches: " + audioBatches);
            System.out.println("Sensor Batches: " + sensorBatches);

        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "An unexpected error occurred.", e);
        }
    }
    private static String getUserTextInput() {
        // Use JOptionPane to show a text input dialog
        return JOptionPane.showInputDialog(null, "Enter text for preprocessing here:");
    }

    private static String arrayToString(double[] array) {
        StringBuilder builder = new StringBuilder("[");
        for (double value : array) {
            builder.append(value).append(", ");
        }
        if (builder.length() > 1) {
            builder.setLength(builder.length() - 2);
        }
        builder.append("]");
        return builder.toString();
    }
}
