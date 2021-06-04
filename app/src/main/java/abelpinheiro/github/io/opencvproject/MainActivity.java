package abelpinheiro.github.io.opencvproject;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.calib3d.StereoSGBM;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.ClipData;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.view.SurfaceView;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_64FC1;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class MainActivity extends Activity /*implements OnTouchListener, CvCameraViewListener2*/ {
    private static final String  TAG              = "MainActivity";

    private ImageView imageView;
    private Mat imageMat;
    private ArrayList<Mat> arrayImageMat = new ArrayList<>();
    static final int REQUEST_IMAGE_CAPTURE = 0;
    static final int REQUEST_IMAGE_GALLERY = 1;
    final int ROWS = 34;
    final int COLS = 3;
    final int PLANES = 1;

    private Mat                                 K;              // MATRIX
    private Mat                                 dist;           //
    private double                              ret;            //
    private double                              focal_length;   //
    Mat                                         rvecs;          // VETOR DE ROTAÇÃO
    Mat                                         tvecs;          // VETOR DE TRANSLAÇÃO

    private CameraBridgeViewBase mOpenCvCameraView;

    /*
    * LOADER PARA CARREGAR A LIB DO OPENCV NO APP. SÓ É POSSÍVEL EXECUTAR AS FUNÇÕES DA LIB
    * DEPOIS DO LOADER SER CARREGADO
    * */
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    getCameraParams();
                    //mOpenCvCameraView.enableView();
                    //mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        //requestWindowFeature(Window.FEATURE_NO_TITLE);
        //getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imagem);


       // mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);
       // mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        //mOpenCvCameraView.setCvCameraViewListener(this);
    }

    /*
    * FUNÇÃO PARA LER O ARQUIVO PARAMS.JSON
    * */
    public String loadJSONFromAsset() {
        String json = null;
        try {
            InputStream is = getAssets().open("params.json");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, "UTF-8");
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json;
    }

    private ArrayList<Mat> getDoubleFromString(JSONArray jsonArray) throws JSONException {
        ArrayList<Mat> res = new ArrayList<>();
        double[] arr;
        for (int i = 0; i < jsonArray.length(); i++) {
            String row = jsonArray.getString(i);
            Log.e(getPackageName(), "ROW É : " + row);
            String[] tokens = row.split("\\[|,|\\]");
            double[] row_double = new double[tokens.length - 1];
            //arr new double[tokens.length - 1];
            for (int j = 1; j < tokens.length; j++){
                row_double[j - 1] = Double.parseDouble(tokens[j]);
            }

            //arr.add(row_double);
            Mat m = new Mat(1, tokens.length - 1, CV_32F );
            m.put(0,0, row_double);
            res.add(m);
        }
        return res;
    }

    /*
    * FUNÇÃO PARA OBTER OS PARAMETROS DO ARQUIVO PARAMS.JSON NA PASTA ASSETS
    * PELA FALTA DE TEMPO OS PARAMETROS SÃO OBTIDOS HARDCODED
    * */
    private void getCameraParams() {
        try {
            JSONObject obj = new JSONObject(loadJSONFromAsset());
            //Log.e(getPackageName(), "VALOR DO JSON É \n\n" + obj);
            ret = obj.getDouble("ret");
            focal_length = obj.getDouble("focal_length");
            JSONArray K_array = obj.getJSONArray("K");
            JSONArray dist_array = obj.getJSONArray("dist");
            JSONArray rvecs_array = obj.getJSONArray("rvecs");
            JSONArray tvecs_array = obj.getJSONArray("tvecs");

            double[] k_data = {3126.384003803299, 0.0, 1993.8432404884418,
                    0.0, 3125.5718243324823, 1414.9908985007826,
                    0.0, 0.0, 1.0};
            K = new Mat(3, 3, CV_32F);
            K.put(0,0, k_data);

            double[] dist_data = {0.14294342139821864, -0.48042444740266055,
                    -0.006612402628681109, 0.001327060089753498, 0.4950991956937261};
            dist = new Mat(1, 5, CV_32F);
            dist.put(0,0, dist_data);

            double[][] rvecs_data = {{-0.00112228, -0.48988971, 0.00173007}, {-0.11021983, 0.27311065, 0.02445877}, {-0.10359594, -0.32621366, 0.00470574},
                    {-0.06559684, -0.08061009, 0.00115442}, {-0.06107416, -0.07174593, -0.02799705}, {-0.11523474, -0.28393419, -0.00677324},
                    {-0.24031959, -0.04092434, -0.00978026}, {-0.12164979, -0.34963914, -0.02569733}, {-0.07672903, 0.01970392, 0.01742422},
                    {-0.05213495, 0.25230858, 0.03655516}, {-0.0571421, -0.36781867, -0.03738548}, {-0.05823772, -0.12183579, -0.0429007},
                    {-0.0640274, -0.61908173, -0.04071412}, {-0.09813366, -0.47693747, -0.05922179}, {-0.00611242, -0.25415615, -0.0375141},
                    {-0.12194974, -0.31744519, -0.00204052}, {0.05774466, 0.2378108, 0.01176283}, {0.0372934, -0.03996528, -0.02212389},
                    {0.01842294, 0.10926666, 0.01194773}, {-0.15404242, -0.03220221, -0.00356754}, {0.01539101, -0.25446608, -0.08090708},
                    {-0.08227524, -0.29641416, -0.05455928}, {-0.14976721, -0.3864943, -0.01353603}, {0.27752243, -0.1556456, -0.00043559},
                    {-0.22927449, -0.56864323, -0.02803492}, {-0.11702286, -0.55371453, -0.04598093}, {-0.31427807, -0.12658386, -0.02965755},
                    {-0.1885246, 0.12059606, 0.0689625}, {-0.19787263, 0.09086841, 0.06147752}, {-0.06523001, -0.45722961, -0.03815495},
                    {0.00740631, 0.22376502, 0.00899018}, {0.00116032, -0.37286579, -0.04064825}, {0.11248238, 0.03712752, -0.05117861}, {-0.19270958, 0.09347687, 0.05798616}};

            rvecs = new Mat(34, 3, CV_32FC1);
            for(int i = 0; i < ROWS; i++){
                rvecs.put(i, 0, rvecs_data[i]);
            }

            double[][] tvecs_data = {{-2.74433137, -2.43945648, 13.14066526}, {-3.20834195, -0.37785919, 17.87258028}, {-1.79765972, -5.29609913, 18.72377589},
                    {-1.11173968, -4.67941603, 16.64117921}, {-2.18120832, -2.47922807, 18.53167258}, {-8.33118897, 0.90432815, 18.64375984},
                    {-11.2289248, 2.99929894, 22.89022133}, {-4.16782145, -0.50968237, 13.92911438}, {-3.54953294, -2.50084371, 14.90469229},
                    {-2.68438159, -1.78438114, 15.29236711}, {-1.64636601, -2.47957531, 19.02395998}, {-4.45719288, -2.43569868, 18.44856246},
                    {-1.24166229, -3.43951216, 22.92551155}, {-6.51333974, -3.83625576, 13.28188062}, {-0.74584174, -4.09522064, 15.5774366},
                    {-0.7237245, -4.14888284, 16.1770353}, {-1.74770949, -5.46214339, 20.15624387}, {1.8183736, -6.11506435, 19.38462574},
                    {-8.10038952, -2.9361873, 24.0753327}, {1.50938178, 2.29359513, 21.470287}, {-8.15333112, -4.96833656, 17.34575779},
                    {-2.59637515, -1.34576083, 15.33683228}, {-2.07631564, -2.29250115, 20.38555434}, {-3.7378349, -3.42182113, 14.05404275},
                    {-4.0028294, -1.73493614, 13.69534659}, {-4.864757, -4.04031086, 13.67063586}, {-3.30509827, -0.83859246, 16.03191564},
                    {-6.19356254, -2.33911319, 18.07532695}, {-5.9015336, -2.0489583, 16.15980973}, {-5.60779956, -4.41538879, 14.21922866},
                    {-1.4947524, -4.80732363, 17.46601003}, {-2.4990293, -4.56833117, 15.33671766}, {-12.62846038, -7.36920862, 21.26903899}, {-3.74055554, -0.5005747, 16.98682451}};

            tvecs = new Mat(34, 3, CV_32FC1);
            for(int i = 0; i < ROWS; i++){
                tvecs.put(i, 0, tvecs_data[i]);
            }

            Log.e(getPackageName(), "RET É: " + ret);
            Log.e(getPackageName(), "VALOR DE RVECS É: " + rvecs.dump());
            Log.e(getPackageName(), "VALOR DE TVECS É: " + tvecs.dump());
            Log.e(getPackageName(), "VALOR DE K É: " + K.dump());
            Log.e(getPackageName(), "VALOR DE DIST É: " + dist.dump());


            /*K = getDoubleFromString(K_array);
            for (Mat c : K){
                Log.e(getPackageName(), "RESULTADO CDUMP DA MATRIZ K : " + c.dump());
            }
            dist = getDoubleFromString(dist_array);

            for (Mat c : dist){
                Log.e(getPackageName(), "RESULTADO CDUMP DA MATRIZ DIST: " + c.dump());
            }

            /*for (int i = 0; i < rvecs_array.length(); i++){
                String row = K_array.getString(i);
                ArrayList<Double> row_double = getDoubleFromString(row);
                rvecs.add(row_double);
            }

            //RVECS E TVECS N TA PEGANDO
            Log.e(getPackageName(), "rvecs: " + rvecs);

            for (int i = 0; i < tvecs_array.length(); i++){
                String row = K_array.getString(i);
                ArrayList<Double> row_double = getDoubleFromString(row);
                tvecs.add(row_double);
            }
            Log.e(getPackageName(), "tvecs: " + tvecs);*/


        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    /*
    * FUNÇÃO PARA FAZER A RECONSTRUÇÃO 3D DE 2 IMAGENS
    * */
    private void reconstruct3DImage(Size imageSize){
        // Obter a matriz da camera otimizada
        Mat newCameraMatrix = Calib3d.getOptimalNewCameraMatrix(K, dist, imageSize, 1);
        Log.e(getPackageName(), "NEW CAMERA MATRIZ = " + newCameraMatrix.dump());
        Log.e(getPackageName(), "NUMERO DE DISPARIDADES: " + imageSize);
        // Matrizes das fotos obtidas
        Mat left = arrayImageMat.get(0);
        Mat right = arrayImageMat.get(1);

        // Matrizes finais que receberam o processamento do algoritmo
        Mat destination = new Mat();
        Mat destination2 = new Mat();

        // Conversão para escala de cinza e armazenando em destination/destination2
        Imgproc.cvtColor(left, destination, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(right, destination2, Imgproc.COLOR_RGB2GRAY);

        // Matrizes que receberam as imagens após o undistort
        //TODO Calib3d.triangulatePoints(); OPCIONAL, VER DEPOIS
        Mat dst = new Mat(imageSize, CV_8U);
        Mat dst2 = new Mat(imageSize, CV_8U);

        // Remoção das distorções nas duas imagens
        //Calib3d.undistortImage(destination,dst, newCameraMatrix, dist);
        Imgproc.undistort(destination, dst, K, dist, newCameraMatrix);
        Imgproc.undistort(destination2, dst2, K, dist, newCameraMatrix);

        // Obtém o mapa de disparidade das duas fotos
        Mat disparity = getDisparityMap(dst, dst2);

        Mat Q = new Mat(4,4, CV_32F);
        Mat R1 = new Mat();
        Mat R2 = new Mat();
        Mat P1 = new Mat();
        Mat P2 = new Mat();
        Mat _3dImage = new Mat();
        //Calib3d.stereoRectify(K, dist, K, dist, imageSize, rvecs, tvecs, R1, R2, P1, P2, Q);
        Log.e(getPackageName(), "Q é " + Q);
        double[] Q_data = {1, 0, 0, imageSize.width * -0.5,
                0, -1, 0, imageSize.height * 0.5,
                0, 0, 0, -focal_length,
                0, 0, 1, 0};
        Q.put(0,0, Q_data);
        Calib3d.reprojectImageTo3D(disparity, _3dImage, Q);

        Log.e(getPackageName(), "3d image: " + _3dImage);

        // Obtendo a imagem final e exibindo numa view
        Bitmap bmp = null;
        bmp = Bitmap.createBitmap(_3dImage.cols(), _3dImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(_3dImage, bmp);
        imageView.setRotation(90);
        imageView.setImageBitmap(bmp);

    }

    /*
    * FUNÇÃO PARA O CALCULO DO MAPA DE DISPARIDADE
    * TODO: VERIFICAR SE É PRECISO MELHORAR A TUNAGEM DOS PARAMETROS
    * */
    private Mat getDisparityMap(Mat left, Mat right) {
        Mat disparity = new Mat(left.size(), left.type());

        int numDisparity = (int) (left.size().width/8);
        Log.e(getPackageName(), "NUMERO DE DISPARIDADES: " + numDisparity + "TAMANHO DA IMAGEM: " + left.size() + "LARGURA: " + left.size().width);

        StereoSGBM stereoAlgo = StereoSGBM.create(
                0,    // min DIsparities
                128 , // numDisparities
                11,   // SADWindowSize
                2*11*11,   // 8*number_of_image_channels*SADWindowSize*SADWindowSize   // p1
                5*11*11,  // 8*number_of_image_channels*SADWindowSize*SADWindowSize  // p2

                -1,   // disp12MaxDiff
                63,   // prefilterCap
                10,   // uniqueness ratio
                0, // sreckleWindowSize
                32, // spreckle Range
                0); // full DP

        stereoAlgo.compute(left, right, disparity);
        Log.e(getPackageName(), "Executou a função stereoAlgo.compute");

        Core.normalize(disparity, disparity, 0, 256, Core.NORM_MINMAX, CV_8U);
        Log.e(getPackageName(), "Normalizou a disparidade");
        return disparity;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    /*
    * OPÇÃO USAR A CAMERA NÃO ESTÁ FUNCIONANDO. SELECIONAR ELA VAI CRASHAR O APP
    * */
    public void selecionarFoto(View view) {
        final CharSequence[] options = { "Usar a câmera", "Pegar da galeria","Cancelar" };

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Obter uma irmagem");

        builder.setItems(options, new DialogInterface.OnClickListener() {

            @Override
            public void onClick(DialogInterface dialog, int item) {

                if (options[item].equals("Usar a câmera")) {
                    Intent takePicture = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                    if (takePicture.resolveActivity(getPackageManager()) != null) {
                        startActivityForResult(takePicture, REQUEST_IMAGE_CAPTURE);
                    }

                } else if (options[item].equals("Pegar da galeria")) {
                    Intent pickPhoto = new Intent();
                    pickPhoto.setType("image/*");
                    pickPhoto.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                    pickPhoto.setAction(Intent.ACTION_GET_CONTENT);
                    startActivityForResult(Intent.createChooser(pickPhoto, "Select Picture") , REQUEST_IMAGE_GALLERY);

                } else if (options[item].equals("Cancelar")) {
                    dialog.dismiss();
                }
            }
        });
        builder.show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case REQUEST_IMAGE_CAPTURE:
                    if (resultCode == RESULT_OK && data != null) {
                        Bitmap thumbnail = data.getParcelableExtra("data"); // PEGA A THUMBNAIL da imagem, versão de tamanho reduzido
                        Utils.bitmapToMat(thumbnail, imageMat);
                        Log.e(getCallingPackage(), "objeto é: " + imageMat);
                        imageView.setImageBitmap(thumbnail);
                    }
                    break;
                case REQUEST_IMAGE_GALLERY:
                    if (resultCode == RESULT_OK && data != null) {
                        ArrayList<Uri> mArrayUri = new ArrayList<>();

                        if (data.getClipData() != null){
                            //ClipData mClipData = data.getClipData();
                            int cout = data.getClipData().getItemCount();
                            for (int i = 0; i < cout; i++){
                                Uri imageurl = data.getClipData().getItemAt(i).getUri();
                                mArrayUri.add(imageurl);
                            }

                            try {
                                for (int j = 0; j < mArrayUri.size(); j++){
                                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), mArrayUri.get(j));
                                    Mat m = new Mat();
                                    Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                                    Utils.bitmapToMat(bmp32, m);
                                    arrayImageMat.add(m);
                                }
                                /*Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), mArrayUri.get(0));
                                imageMat = new Mat();
                                Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                                Utils.bitmapToMat(bmp32, imageMat);*/
                                //Log.d(getCallingPackage(), imageMat.toString() + "\nrows" + imageMat.rows() + "\nsize" + imageMat.size() + "\nwidth" + imageMat.width());
                                reconstruct3DImage(arrayImageMat.get(0).size());
                            } catch (IOException e) {
                                e.printStackTrace();
                            }

                        }else {
                            Toast.makeText(this, "É preciso selecionar duas imagens", Toast.LENGTH_SHORT).show();
                            imageView.setImageDrawable(null);
                        }
                    }
                    break;
            }
        }
    }

    /*
     * FUNÇÃO NÃO ESTÁ SENDO UTILIZADA. CALIBRAÇÃO FOI EXECUTADA EM PYTHON E O ARQUIVO COM OS PARAMETROS
     * ESTÁ ARMAZENADO LOCALMENTE
     * */
    public void calibrate(/*List<Mat> pTrainingImages*/){
        // Encontrar os cantos do tabuleiro de xadrez
        Mat grayImage = new Mat();
        Imgproc.cvtColor(imageMat, grayImage, Imgproc.COLOR_RGB2GRAY); // conversão da imagem para escala de cinza
        Size imageSize = null;

        // board height e width é dependente da imagem de tabuleiro utilizada! se for diferente, não vai ser encontrado
        int boardH = 6, boardW = 9; // Altura do tabuleiro, Comprimento do tabuleiro
        Size boardSize = new Size(boardW, boardH);
        MatOfPoint2f corners = new MatOfPoint2f();

        List<Mat>imgPoints = new ArrayList<>(); // 3D point in real world space
        List<Mat>objPoints = new ArrayList<>(); // 2D points in image plane

        //for(Mat img : pTrainingImages) {} LOOP DE MUTIPLAS IMAGENS

        imageSize = imageMat.size();
        int boardN = boardH * boardW; // Quantidade de pixels na imagem

        MatOfPoint3f obj = new MatOfPoint3f();

        for (int j=0; j< boardN; j++)
        {
            obj.push_back(new MatOfPoint3f(new Point3((double)j/(double)boardW, (double)j%(double)boardH, 0.0d)));
        }

        objPoints.add(obj);


        boolean found_chess = Calib3d.findChessboardCorners(grayImage, boardSize, corners,Calib3d.CALIB_CB_ADAPTIVE_THRESH + Calib3d.CALIB_CB_NORMALIZE_IMAGE + Calib3d.CALIB_CB_FAST_CHECK );

        if (found_chess){
            //Se encontrou, calibrar os parâmetros
            TermCriteria term = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.001); // definir um critério para precisão do pixel
            Imgproc.cornerSubPix(grayImage, corners, new Size(5, 5), new Size(-1, -1), term); // refinar a localização dos corners baseado no critério

            //Exibir o tabuleiro
            Calib3d.drawChessboardCorners(imageMat, boardSize, corners, found_chess);
            Bitmap newBmp = Bitmap.createBitmap(grayImage.width(), grayImage.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(imageMat, newBmp);
            //imageView.setImageBitmap(newBmp);

            imgPoints.add(corners);

            performCalibration(objPoints, imgPoints, imageSize);
        }
    }

    /*
    * FUNÇÃO NÃO ESTÁ SENDO UTILIZADA. CALIBRAÇÃO FOI EXECUTADA EM PYTHON E O ARQUIVO COM OS PARAMETROS
    * ESTÁ ARMAZENADO LOCALMENTE
    * */
    private void performCalibration(List<Mat> objPoints, List<Mat> imgPoints, Size imageSize) {
        Mat cameraMatrix = new Mat(3,3, CV_32FC1); // matrix da camera K
        Mat distCoeffs = new Mat(); // coeficientes de distorção

        List<Mat> rvecs = new ArrayList<>(); // vetor de rotação
        List<Mat> tvecs = new ArrayList<>(); // vetor de translação
        Calib3d.calibrateCamera(objPoints, imgPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

        Log.d("camera", "VALOR DE CAMERAMATRIX: " + cameraMatrix + "\nVALOR DE DISCOEFFS: " + distCoeffs);
        Log.d("camera", "distcoeffs: " + distCoeffs.dump());
        Log.d("camera", "camera matrix: " + cameraMatrix.dump());
       // Log.d("camera", "rotation vec: " + rvecs.);
        //Log.d("camera", "translation vec: " + cameraMatrix.dump());

        Mat undistorted = new Mat();
        Imgproc.undistort(imageMat, undistorted, cameraMatrix, distCoeffs);
        Bitmap tempBmp = Bitmap.createBitmap(imageMat.width(), imageMat.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageMat, tempBmp);
        imageView.setImageBitmap(tempBmp);
    }
}