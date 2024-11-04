#include <math.h> 
 #include <string.h> 
#include "./include/k2c_include.h" 
#include "./include/k2c_tensor_include.h" 

 


void foobar(k2c_tensor* keras_tensor_8_input, k2c_tensor* keras_tensor_9_output) { 

float lstm_fwork[80] = {0}; 
int lstm_go_backwards = 0;
int lstm_return_sequences = 0;
float lstm_state[20] = {0}; 
float lstm_kernel_array[320] = {
+8.01765323e-02f,+1.92417651e-01f,-1.60194859e-01f,-1.37521207e-01f,+3.11533064e-01f,
+1.72925889e-02f,-1.83737323e-01f,-1.18413672e-01f,+2.80161798e-02f,-2.90687829e-01f,
-1.10218227e-01f,-2.29733527e-01f,+2.90815443e-01f,+1.89860195e-01f,+2.40151197e-01f,
+2.89492279e-01f,-6.78732693e-02f,+2.67122060e-01f,-1.49875879e-04f,-8.06239843e-02f,
+3.21006477e-02f,-3.51072550e-01f,-1.48431763e-01f,+3.92983258e-02f,-1.11919776e-01f,
-4.39700484e-02f,-9.59365070e-02f,+3.08204979e-01f,-6.42592609e-02f,+3.15774709e-01f,
-2.79670298e-01f,+8.72162282e-02f,+8.23453367e-02f,+3.88357937e-02f,-1.53995812e-01f,
+6.11797869e-02f,+2.90438384e-01f,+3.06908637e-01f,-8.82261693e-02f,-6.50568604e-02f,
+3.08743119e-03f,-9.45925713e-03f,+1.94046110e-01f,+2.56480187e-01f,-1.67021811e-01f,
-2.44656056e-01f,+1.77004904e-01f,+3.59368324e-02f,-2.07651287e-01f,-2.95984089e-01f,
+2.91492969e-01f,+2.69748420e-01f,-5.63055873e-02f,-8.80782306e-02f,-3.50842923e-01f,
-2.72853315e-01f,-3.18977207e-01f,-9.10156071e-02f,+3.84224057e-02f,+2.28653818e-01f,
-2.98755467e-02f,+1.61381572e-01f,-3.18311214e-01f,-2.72095591e-01f,-3.47687989e-01f,
-1.30336091e-01f,-2.01897621e-02f,-2.10161969e-01f,+8.80016983e-02f,-2.99783707e-01f,
-2.25570187e-01f,+8.77732038e-03f,-8.11578035e-02f,+1.12005591e-01f,-6.61872327e-02f,
+1.42660767e-01f,+5.23562431e-02f,-3.27090144e-01f,-2.12348208e-01f,-9.29794908e-02f,
-1.69186890e-01f,-1.26707330e-01f,+2.24276692e-01f,-2.87255943e-02f,-2.16269389e-01f,
-1.91732168e-01f,-3.26673388e-02f,+1.86352640e-01f,-1.19292095e-01f,-4.36988771e-02f,
+5.82318604e-02f,-2.96189487e-01f,+2.03375965e-01f,+2.90482551e-01f,-2.60944039e-01f,
-9.27978456e-02f,+9.46747959e-02f,-9.91871357e-02f,+2.43305773e-01f,+5.36827743e-02f,
+2.73210675e-01f,-9.54725444e-02f,-3.25432003e-01f,+3.09512109e-01f,-2.06511036e-01f,
+4.30823267e-02f,+1.08971179e-01f,-2.84950972e-01f,+2.18817860e-01f,+1.93327576e-01f,
+2.94116825e-01f,+2.26803392e-01f,-4.39282358e-02f,-1.30668461e-01f,+3.03551227e-01f,
-1.62030190e-01f,+2.04929084e-01f,-2.79021829e-01f,-2.09954947e-01f,+2.30375320e-01f,
-1.72317475e-01f,-5.74284792e-03f,-3.49684149e-01f,+2.84673184e-01f,-2.87534297e-01f,
+6.31744266e-02f,+1.42420858e-01f,+1.55689716e-02f,-1.32076174e-01f,-3.26008230e-01f,
-1.77787036e-01f,+2.77554184e-01f,+2.96621829e-01f,+2.11413503e-02f,-1.97918221e-01f,
-1.01164818e-01f,-5.14349937e-02f,-1.60363197e-01f,-1.85900137e-01f,-2.65194803e-01f,
+3.81982028e-02f,+2.86395818e-01f,-6.08845055e-02f,+6.53876364e-02f,+2.39078730e-01f,
-2.74587214e-01f,-2.16191158e-01f,-4.27749753e-03f,+3.20138603e-01f,-1.94039032e-01f,
+8.73135924e-02f,+3.07842284e-01f,-4.00757790e-03f,+1.12447292e-01f,+8.80959332e-02f,
-1.96368992e-02f,+7.05753267e-02f,-5.89036644e-02f,-2.30309680e-01f,-3.23269010e-01f,
-4.20400500e-02f,-9.45543349e-02f,-1.03531718e-01f,+2.75870651e-01f,-2.70975411e-01f,
+1.80879086e-01f,-1.50306448e-01f,+3.27288240e-01f,-1.22006103e-01f,+1.96688622e-01f,
-3.33816350e-01f,+1.36872888e-01f,-2.17512459e-01f,+2.55870968e-01f,-2.08559468e-01f,
-2.57898271e-02f,-3.52216810e-01f,+2.61321217e-01f,-3.00459743e-01f,+5.92665672e-02f,
-2.29907766e-01f,+1.57166034e-01f,-2.12882131e-01f,-2.94301480e-01f,+2.37827092e-01f,
+2.00818747e-01f,-2.83395737e-01f,-1.25025004e-01f,+1.87214345e-01f,-9.82981920e-03f,
+2.28091985e-01f,-2.64934570e-01f,-3.84229124e-02f,-1.14806071e-01f,-2.55972385e-01f,
-5.86932003e-02f,-8.16738605e-02f,-1.50189698e-01f,+2.79646784e-01f,+3.48610073e-01f,
-2.80822754e-01f,+7.63337612e-02f,+2.40197331e-01f,-1.08693510e-01f,-4.03618515e-02f,
-3.44233692e-01f,+1.99654013e-01f,-2.25364164e-01f,-3.38094503e-01f,+3.09688717e-01f,
-1.49222091e-01f,-1.15610749e-01f,+3.15761596e-01f,+1.02514207e-01f,+8.55194926e-02f,
+3.53312999e-01f,+3.11858505e-01f,-2.80475795e-01f,+3.52028102e-01f,-2.33122975e-01f,
+1.44640148e-01f,-3.29327285e-01f,-3.43344152e-01f,-1.35230944e-01f,+3.22623342e-01f,
+2.74707943e-01f,+3.29317957e-01f,-1.04587063e-01f,-3.14687580e-01f,-1.73651591e-01f,
+7.32309818e-02f,+1.18566155e-01f,-2.38973305e-01f,+1.45230711e-01f,+1.84743792e-01f,
-3.13431859e-01f,+1.20620906e-01f,-1.92103148e-01f,+3.50192934e-01f,-3.42957586e-01f,
-1.88804567e-01f,-1.89421341e-01f,-9.29618776e-02f,-1.38900757e-01f,+2.74959058e-01f,
-1.86140463e-01f,-6.02301359e-02f,+5.25527298e-02f,-1.14724040e-03f,+4.19246554e-02f,
+2.62826473e-01f,-2.01272696e-01f,-5.46290576e-02f,+2.77943939e-01f,+6.22857213e-02f,
+1.59580797e-01f,-1.57484218e-01f,-1.35156006e-01f,+2.90112942e-01f,-1.61646813e-01f,
-2.48776406e-01f,-5.26401401e-02f,-7.30466545e-02f,-6.03687763e-02f,-3.41142893e-01f,
+2.70270854e-01f,-2.87431210e-01f,+3.33647132e-02f,+1.73701137e-01f,-2.03410089e-02f,
+1.53695911e-01f,-3.95774841e-02f,-1.88163921e-01f,-5.11461496e-03f,+2.49321967e-01f,
+9.14867222e-02f,+2.24874169e-01f,+1.61575645e-01f,+1.04466945e-01f,+2.82151729e-01f,
-5.40374815e-02f,-2.47507274e-01f,-1.14766955e-01f,+1.05797857e-01f,-1.07563645e-01f,
-2.21829221e-01f,-1.71074301e-01f,-1.43071771e-01f,-2.59854108e-01f,+2.14350969e-01f,
+6.40468597e-02f,+1.94298834e-01f,+7.29560852e-03f,-2.35865980e-01f,-2.91292369e-01f,
-2.87257731e-01f,-2.15121388e-01f,-5.75650036e-02f,+2.27737099e-01f,+3.34834963e-01f,
-8.95111263e-02f,-8.45240653e-02f,-6.53406084e-02f,-2.02244014e-01f,-7.06611276e-02f,
-1.03019714e-01f,-1.37052864e-01f,-1.56837523e-01f,+1.18350983e-02f,+5.83059490e-02f,
-2.14354575e-01f,-3.18950415e-03f,-3.48794162e-01f,+9.79255736e-02f,-1.60196200e-01f,
-2.73867548e-02f,+3.40398759e-01f,-1.64116621e-01f,-5.49862981e-02f,+1.76141530e-01f,
}; 
k2c_tensor lstm_kernel = { &lstm_kernel_array[0], 2, 320, { 32,10, 1, 1, 1 } }; 
float lstm_recurrent_kernel_array[400] = {
-5.78502417e-02f,-1.51433900e-01f,+3.79074097e-01f,+1.07813589e-01f,-4.29098234e-02f,
-1.21875189e-01f,+1.42348692e-01f,+8.88927653e-02f,+2.28865728e-01f,+2.54745960e-01f,
-1.13204010e-02f,-1.59202114e-01f,+3.59465182e-02f,+1.06292859e-01f,+2.55693763e-01f,
-1.39307985e-02f,-2.30242640e-01f,+1.05080232e-01f,+1.67379193e-02f,-2.76456326e-01f,
+3.99377346e-01f,+2.28951499e-02f,+8.68620425e-02f,+2.44351298e-01f,-5.33764660e-02f,
-1.66246817e-01f,-1.77024826e-02f,-1.79360181e-01f,+1.59460604e-01f,-1.15919188e-02f,
+6.07865267e-02f,-1.07585602e-01f,+2.16178931e-02f,-2.09277831e-02f,-1.83937594e-01f,
+1.70072183e-01f,-3.34756494e-01f,+6.62132949e-02f,-1.46309435e-01f,+1.34846225e-01f,
+2.34843791e-01f,-5.16590513e-02f,-8.54302198e-02f,+1.09851211e-01f,+1.24360986e-01f,
+1.99493915e-01f,-1.46899596e-02f,-9.07610059e-02f,+1.49514154e-01f,+9.95340198e-03f,
-8.60316828e-02f,+3.67972814e-03f,-1.79717407e-01f,-3.66396047e-02f,+4.10255432e-01f,
+1.75896868e-01f,+9.20720398e-02f,+1.45223171e-01f,+7.15110973e-02f,+1.72563821e-01f,
+8.63362849e-02f,+2.06185967e-01f,-4.46953997e-02f,+2.67483324e-01f,+1.32166341e-01f,
-1.52423568e-02f,+1.00952521e-01f,-2.54916161e-01f,+1.91165432e-01f,-1.02606341e-01f,
-1.82582691e-01f,+6.18120022e-02f,-2.77295500e-01f,-3.78229842e-03f,+1.26711980e-01f,
-4.11590964e-01f,+1.25401959e-01f,+6.76614642e-02f,+4.10831124e-02f,-5.43925017e-02f,
-1.19648948e-01f,-1.33310914e-01f,-6.93254247e-02f,+6.06551766e-02f,-1.93601862e-01f,
+2.12017633e-02f,+1.30228117e-01f,-2.17205167e-01f,+2.45376274e-01f,-4.71053645e-02f,
+1.92142844e-01f,-8.64798054e-02f,-1.33213639e-01f,+6.06252924e-02f,+1.01111814e-01f,
+1.48575872e-01f,-4.76196744e-02f,-1.38785228e-01f,+2.97817551e-02f,-2.15988204e-01f,
-1.12558417e-01f,-1.40763225e-03f,+1.33879676e-01f,+8.63212943e-02f,+1.55297786e-01f,
-9.85963196e-02f,-1.76155776e-01f,-2.38535285e-01f,-1.36815419e-03f,+4.52012382e-02f,
-3.89928594e-02f,-1.57603890e-01f,+7.63536170e-02f,+8.03249702e-03f,-1.50781706e-01f,
-4.96727340e-02f,+7.50593692e-02f,-8.94493461e-02f,-1.87712535e-02f,+1.61234792e-02f,
-3.80632281e-02f,+7.44707212e-02f,-3.07355374e-01f,+1.21755973e-01f,+4.90919463e-02f,
-6.28885850e-02f,+1.13562107e-01f,-9.32750553e-02f,-1.13145113e-01f,-2.15381369e-01f,
+6.08356856e-02f,+2.51847446e-01f,-1.67761266e-01f,+3.90370004e-02f,-2.78804451e-01f,
-9.11940560e-02f,-3.84460762e-02f,-8.62956047e-02f,-8.29924718e-02f,-1.49537774e-03f,
-3.96943316e-02f,+1.15133405e-01f,-4.00226936e-02f,-6.78725541e-04f,+3.27372193e-01f,
+2.85311699e-01f,-1.19103312e-01f,+7.57792145e-02f,+2.23323762e-01f,-9.33675561e-03f,
+1.74919330e-03f,+1.84531525e-01f,-3.37303355e-02f,-8.88712853e-02f,+2.00597882e-01f,
+8.58610123e-02f,-2.06141427e-01f,-1.18493937e-01f,-7.31149837e-02f,-1.94809392e-01f,
-1.09383978e-01f,-1.87284410e-01f,-7.87468478e-02f,-1.90027822e-02f,-1.53912991e-01f,
-6.88879490e-02f,-1.35062069e-01f,+6.92948401e-02f,-2.49535903e-01f,+1.52203254e-02f,
+2.31548592e-01f,-9.71304700e-02f,+9.13279951e-02f,+1.04259253e-02f,+3.70639116e-02f,
+1.40968874e-01f,-8.51456448e-02f,+5.90738766e-02f,+2.07634550e-02f,-2.84301907e-01f,
+1.31646283e-02f,+1.01887789e-02f,+4.50730622e-02f,+2.02856779e-01f,+1.11543097e-01f,
+9.29635689e-02f,+2.95658968e-02f,-1.91241667e-01f,+1.10193767e-01f,-1.77832752e-01f,
-1.56698093e-01f,-2.92275816e-01f,+7.42857978e-02f,-1.51761547e-01f,+4.20780741e-02f,
+2.84100950e-01f,-1.00409910e-02f,-2.36576736e-01f,-3.82097326e-02f,+2.48397335e-01f,
+1.79439515e-01f,-6.33354485e-02f,-1.51791021e-01f,+3.31467250e-03f,+1.60645321e-01f,
-2.35678494e-01f,+2.89434433e-01f,-2.03729883e-01f,-4.75013740e-02f,+1.32337764e-01f,
-9.34592336e-02f,-2.86987245e-01f,+3.13172251e-01f,+1.75342321e-01f,+4.71163243e-02f,
-1.36903692e-02f,+3.94115567e-01f,-2.16491804e-01f,+2.43627548e-01f,+4.54511344e-02f,
+1.43763795e-02f,-1.47912577e-01f,-2.68273622e-01f,+3.70301045e-02f,+1.33298367e-01f,
+5.75126968e-02f,-2.69838423e-02f,-1.76641956e-01f,+1.17144808e-01f,+5.07914536e-02f,
+2.30351225e-01f,+7.83193633e-02f,+1.12693623e-01f,-4.27054495e-01f,-2.44865835e-01f,
+1.99776590e-02f,+6.11569360e-02f,-2.11096108e-01f,-3.29766162e-02f,+3.53962928e-01f,
-2.48772115e-01f,-2.92260766e-01f,+3.89511697e-03f,-2.65124589e-01f,-3.24903876e-01f,
-9.40299481e-02f,+7.93869942e-02f,-3.28588933e-02f,-1.35266259e-01f,-7.11171329e-02f,
+1.77987516e-01f,+1.93138465e-01f,+1.05741441e-01f,+1.23557366e-01f,+3.25219721e-01f,
-7.78019875e-02f,+7.59326220e-02f,+4.28304635e-02f,-1.42874971e-01f,+8.23235810e-02f,
-9.99295618e-03f,+2.56268919e-01f,+1.74991995e-01f,-6.41189283e-03f,-1.51912645e-01f,
-4.03315872e-01f,-1.84123248e-01f,+1.53231286e-02f,+2.67598778e-01f,+1.55173272e-01f,
+8.22070912e-02f,+1.80942684e-01f,+5.70624024e-02f,-3.71288002e-01f,-9.31649283e-02f,
+5.81613705e-02f,+6.54890090e-02f,-1.77386492e-01f,+1.74376026e-01f,-6.48098066e-04f,
-1.36906579e-01f,+6.62746951e-02f,+1.44082338e-01f,-8.06268528e-02f,-1.42435566e-01f,
+1.41132459e-01f,+3.73947442e-01f,+2.45029852e-01f,-5.73320352e-02f,+1.70279130e-01f,
+1.34832859e-01f,+9.42177176e-02f,-9.89427716e-02f,-3.25984180e-01f,+3.61406028e-01f,
+1.85400009e-01f,-1.20416984e-01f,-5.13341539e-02f,-1.26754284e-01f,+1.24375328e-01f,
-6.15333617e-02f,-8.69688913e-02f,-6.01835027e-02f,+1.26529858e-01f,+5.22369891e-02f,
-1.54197842e-01f,+8.58796202e-03f,+2.33329415e-01f,+1.63183302e-01f,+2.64786720e-01f,
-1.96114182e-02f,+2.88756430e-01f,+4.72685918e-02f,+3.78269295e-04f,-2.73798108e-01f,
+1.43180504e-01f,+2.59879772e-02f,-4.61749174e-03f,-1.20628335e-01f,+3.52502763e-02f,
-5.30261621e-02f,-6.17156737e-02f,+1.38284713e-01f,+2.28282005e-01f,+1.52464613e-01f,
+1.66788623e-01f,-7.80936107e-02f,-1.09399781e-01f,-2.77741253e-01f,-3.03127229e-01f,
+1.00350462e-01f,+9.41346586e-02f,-6.52233930e-03f,-7.75188133e-02f,+9.78765637e-02f,
+7.72807300e-02f,-1.66302532e-01f,-2.24703662e-02f,-9.47149750e-03f,+9.09801424e-02f,
-1.26496861e-02f,+2.58341849e-01f,-6.13909438e-02f,+2.46304367e-02f,+1.42573193e-03f,
-2.15961009e-01f,-7.47018009e-02f,+1.77046835e-01f,+1.29626900e-01f,-1.65374890e-01f,
+1.83871254e-01f,+1.32697463e-01f,+6.67630509e-02f,-1.46077305e-01f,+5.46988994e-02f,
+2.38087952e-01f,-2.67111957e-01f,-1.89472139e-02f,-2.44286489e-02f,-2.31224731e-01f,
+1.25792995e-01f,+9.33577791e-02f,+1.50250748e-01f,-2.40563247e-02f,+7.02313706e-02f,
-3.28081436e-02f,+6.67575300e-02f,+4.45451699e-02f,+3.09870809e-01f,+2.82211583e-02f,
-3.34288217e-02f,-9.57463607e-02f,+1.30898831e-02f,+1.12315208e-01f,-2.87047029e-02f,
-9.43575874e-02f,+3.15083228e-02f,+3.04631770e-01f,-3.23719829e-01f,+2.36636549e-02f,
+1.89696938e-01f,-2.37798735e-01f,+2.04817936e-01f,-1.52877107e-01f,+1.49902711e-02f,
+1.61430433e-01f,+2.36050934e-01f,-2.88192570e-01f,+4.98640984e-02f,+3.17887776e-02f,
-1.87690184e-01f,+5.41809015e-02f,-2.73935087e-02f,-8.01409185e-02f,+8.28434005e-02f,
+4.17950340e-02f,+2.15936527e-01f,-4.57045026e-02f,-5.56275062e-02f,+1.73329040e-01f,
}; 
k2c_tensor lstm_recurrent_kernel = { &lstm_recurrent_kernel_array[0], 2, 400, { 40,10, 1, 1, 1 } }; 
float lstm_bias_array[40] = {
+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,
+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,
+1.00000000e+00f,+1.00000000e+00f,+1.00000000e+00f,+1.00000000e+00f,+1.00000000e+00f,
+1.00000000e+00f,+1.00000000e+00f,+1.00000000e+00f,+1.00000000e+00f,+1.00000000e+00f,
+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,
+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,
+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,
+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,+0.00000000e+00f,
}; 
k2c_tensor lstm_bias = { &lstm_bias_array[0], 1, 40, { 40, 1, 1, 1, 1 } }; 

 
k2c_lstm(keras_tensor_9_output,keras_tensor_8_input,lstm_state,&lstm_kernel, 
	&lstm_recurrent_kernel,&lstm_bias,lstm_fwork, 
	lstm_go_backwards,lstm_return_sequences, 
	k2c_selu,k2c_elu); 

 } 

void foobar_initialize() { 

} 

void foobar_terminate() { 

} 

