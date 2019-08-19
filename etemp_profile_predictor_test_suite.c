#include <stdio.h> 
#include <math.h> 
#include <time.h> 
#include "etemp_profile_predictor.h" 

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2);
struct timeval GetTimeStamp(); 
 
int main(){
float test1_input_thomson_temp_EFITRT1_input_array[33] = {
-1.81140418e+00,1.40204668e+00,-1.50484874e+00,-7.69539670e-01,-1.56957479e+00,
6.80786391e-01,-3.23919248e-02,-5.53548324e-01,-1.65857489e+00,-1.67223790e+00,
1.32450999e+00,-1.15305862e-01,-4.55459358e-02,-1.82936837e+00,-1.76921891e-01,
-1.15339583e-01,1.97397993e+00,-6.43957472e-01,3.56021435e-01,-1.63899839e+00,
-2.56218012e-01,1.57061622e+00,6.22015192e-01,1.57441030e+00,2.36751795e-01,
-1.99648231e+00,1.32963338e+00,-3.89509875e-01,-1.17950522e+00,2.37794952e-01,
-7.80639044e-02,8.98616605e-01,6.45099694e-01,}; 
k2c_tensor test1_input_thomson_temp_EFITRT1_input = {&test1_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test1_input_thomson_dens_EFITRT1_input_array[33] = {
1.00648078e+00,4.05144878e-03,-1.00444961e+00,1.71633601e+00,-5.79585867e-01,
1.49450645e+00,-1.53591240e+00,-1.17710648e+00,1.29826658e+00,5.75188446e-01,
-4.38215552e-01,-1.93677661e+00,1.79983215e+00,-1.33076739e-01,2.93626467e-01,
-5.44443744e-01,-7.99570944e-01,-1.75056952e+00,1.44398658e+00,1.89969027e+00,
1.98056276e+00,-1.64283771e+00,1.48402079e+00,1.54504028e+00,-1.75411643e+00,
-1.14404011e+00,1.91318891e+00,-1.35534992e+00,-7.09445974e-01,8.94298340e-01,
7.21442843e-01,1.61488298e+00,-2.13298954e-01,}; 
k2c_tensor test1_input_thomson_dens_EFITRT1_input = {&test1_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test1_input_past_pinj_input_array[8] = {
-1.99915108e+00,-6.87073408e-01,1.30013822e+00,8.01490582e-01,7.96832635e-01,
9.02548292e-01,7.03196044e-01,6.40256109e-01,}; 
k2c_tensor test1_input_past_pinj_input = {&test1_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test1_input_past_curr_input_array[8] = {
-1.64662406e-01,-1.59668552e+00,-7.30002967e-03,-1.99423389e+00,-4.70266753e-01,
-1.87136751e+00,1.94792210e+00,1.88509178e+00,}; 
k2c_tensor test1_input_past_curr_input = {&test1_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test1_input_past_tinj_input_array[8] = {
1.98343679e+00,1.61584792e+00,1.09759091e+00,2.14837301e-01,1.61402163e+00,
6.21819897e-01,1.78966086e+00,-3.82769335e-01,}; 
k2c_tensor test1_input_past_tinj_input = {&test1_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test1_input_past_gasA_input_array[8] = {
-1.40567515e+00,-4.70041039e-01,9.21972373e-01,1.66256116e+00,5.50697869e-01,
-1.43828033e+00,1.72834157e-01,-1.52981169e+00,}; 
k2c_tensor test1_input_past_gasA_input = {&test1_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test1_input_future_pinj_input_array[4] = {
-1.18629498e+00,-1.55689350e+00,1.47485525e+00,-2.01171142e-01,}; 
k2c_tensor test1_input_future_pinj_input = {&test1_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test1_input_future_curr_input_array[4] = {
1.37470570e+00,1.30440648e+00,3.84867987e-01,3.01849616e-02,}; 
k2c_tensor test1_input_future_curr_input = {&test1_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test1_input_future_tinj_input_array[4] = {
-1.35647344e+00,-1.18072774e+00,7.54276214e-01,1.78723551e-01,}; 
k2c_tensor test1_input_future_tinj_input = {&test1_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test1_input_future_gasA_input_array[4] = {
-1.82016495e+00,1.63440664e+00,1.58846505e+00,1.30738370e+00,}; 
k2c_tensor test1_input_future_gasA_input = {&test1_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test1_array[33] = {
2.25732755e-03,-3.92252859e-03,4.66635742e-04,-3.51336435e-04,-4.29938838e-04,
-6.61617320e-04,-1.10656163e-02,-8.45937524e-03,-9.05875489e-03,-9.75561142e-03,
-6.00259844e-03,-2.79296981e-03,-1.90126356e-02,-2.88900118e-02,-3.30761261e-02,
-5.60955741e-02,-6.68904185e-02,-8.30147862e-02,-1.23736396e-01,-1.35759413e-01,
-1.11018375e-01,-1.03364080e-01,-1.23292737e-01,-9.97837186e-02,-6.55548051e-02,
-9.54634026e-02,-1.47368982e-01,-1.42156735e-01,-1.33183867e-01,-8.72549489e-02,
-6.82393238e-02,-7.83551782e-02,-5.41933700e-02,}; 
k2c_tensor keras_target_temp_test1 = {&keras_target_temp_test1_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test1_array[33] = {0}; 
k2c_tensor c_target_temp_test1 = {&c_target_temp_test1_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test1_array[33] = {
2.53505446e-03,1.00342417e-03,4.46707150e-03,2.84784427e-03,1.84862316e-03,
9.90901329e-03,-2.03194141e-01,-1.10008597e-01,-1.74542531e-01,-2.00811177e-01,
-2.17108801e-01,-3.76466095e-01,-4.52323616e-01,-5.21730661e-01,-4.60267961e-01,
-4.97821569e-01,-4.26474571e-01,-4.83307540e-01,-4.35969174e-01,-4.46508288e-01,
-3.10729563e-01,-3.39777887e-01,-2.12980106e-01,-1.02812216e-01,-7.45068118e-03,
-5.14382962e-03,-1.91855859e-02,2.07179319e-03,-2.36995518e-04,-1.61056779e-03,
-2.07831059e-03,4.84677963e-04,-2.38589942e-04,}; 
k2c_tensor keras_target_dens_test1 = {&keras_target_dens_test1_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test1_array[33] = {0}; 
k2c_tensor c_target_dens_test1 = {&c_target_dens_test1_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test2_input_thomson_temp_EFITRT1_input_array[33] = {
-1.32174096e+00,-1.77740085e+00,6.72919254e-01,1.78027992e+00,-1.11839254e+00,
7.34532523e-01,9.93911855e-01,-1.09175033e+00,-8.73882946e-01,-1.21224992e+00,
1.11195649e+00,1.77376711e+00,2.01796538e-01,-1.80022709e+00,1.64585208e+00,
-1.07767981e+00,-5.78320913e-01,1.50391757e+00,-1.93338028e+00,8.30848371e-01,
5.53793194e-02,-2.22685794e-01,-1.47149364e+00,-1.70970897e+00,-5.31220557e-01,
1.40155833e+00,-9.03587201e-03,1.11599390e-01,-1.13233385e+00,1.20970906e+00,
8.08989981e-01,1.04959763e+00,-4.48009913e-01,}; 
k2c_tensor test2_input_thomson_temp_EFITRT1_input = {&test2_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test2_input_thomson_dens_EFITRT1_input_array[33] = {
-1.26919146e+00,2.82033336e-01,1.28811380e+00,1.41666925e+00,-1.31282448e+00,
4.71689227e-01,9.62288666e-01,-7.89627104e-01,-5.15896891e-01,1.69153303e+00,
-1.56655105e+00,-1.14457772e+00,1.58651797e+00,-1.61444838e+00,-3.00174856e-01,
2.36668546e-01,-1.60420551e+00,-1.49319063e+00,1.59604009e+00,-4.70337439e-01,
9.21596139e-01,1.95739056e+00,8.54051922e-01,-6.61115467e-01,1.95628729e+00,
-2.15136980e-01,-3.16866474e-01,-9.55530830e-01,-1.91046373e+00,-2.11490586e-01,
1.98401076e+00,1.65109199e+00,-1.81093134e+00,}; 
k2c_tensor test2_input_thomson_dens_EFITRT1_input = {&test2_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test2_input_past_pinj_input_array[8] = {
-1.70218607e+00,-5.77194447e-01,-1.26607429e+00,-1.81743533e+00,-4.83545155e-02,
1.53986316e+00,5.23956931e-02,-1.48194682e+00,}; 
k2c_tensor test2_input_past_pinj_input = {&test2_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test2_input_past_curr_input_array[8] = {
-1.39990190e+00,1.89967346e+00,-3.58422000e-03,1.80327286e+00,-7.28288519e-01,
2.25102398e-01,8.85428267e-01,-8.94974554e-01,}; 
k2c_tensor test2_input_past_curr_input = {&test2_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test2_input_past_tinj_input_array[8] = {
-3.92344670e-01,1.50662079e+00,1.65946421e+00,-1.88659251e-01,6.84922761e-01,
-1.29371355e+00,-9.88896432e-01,-3.45978477e-01,}; 
k2c_tensor test2_input_past_tinj_input = {&test2_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test2_input_past_gasA_input_array[8] = {
9.77883789e-02,3.43893302e-01,2.15715991e-01,5.06353687e-01,4.43432304e-01,
5.93520529e-01,7.32689924e-01,6.55719346e-02,}; 
k2c_tensor test2_input_past_gasA_input = {&test2_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test2_input_future_pinj_input_array[4] = {
-2.57381345e-02,-9.32335966e-01,-8.22571521e-01,1.10373641e+00,}; 
k2c_tensor test2_input_future_pinj_input = {&test2_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test2_input_future_curr_input_array[4] = {
-3.67923202e-01,1.80180098e+00,-1.21145741e+00,-7.23822006e-01,}; 
k2c_tensor test2_input_future_curr_input = {&test2_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test2_input_future_tinj_input_array[4] = {
1.91315929e-01,-1.59405615e+00,3.94332641e-01,1.60330444e+00,}; 
k2c_tensor test2_input_future_tinj_input = {&test2_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test2_input_future_gasA_input_array[4] = {
-1.89191275e+00,-1.24478741e+00,2.11980809e-01,1.81224403e+00,}; 
k2c_tensor test2_input_future_gasA_input = {&test2_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test2_array[33] = {
-7.79646076e-03,-3.00799077e-03,-1.09249284e-03,-1.54382293e-03,1.64101119e-04,
5.96529280e-04,3.72362410e-04,-1.00828405e-03,-1.04063172e-02,-7.03536998e-03,
-5.93341794e-03,-8.88486952e-03,-1.33393612e-02,-1.32401465e-02,-1.92078166e-02,
5.15271968e-04,1.50819868e-03,-8.38392531e-04,-2.57871160e-03,4.36461996e-03,
1.01533895e-02,-1.03455465e-02,-1.31269870e-03,-1.19900517e-03,-5.76873659e-04,
-1.73712359e-03,-4.25353472e-04,-8.38202657e-04,-2.42922804e-04,2.74654885e-04,
-2.66268733e-04,-2.82333954e-03,8.08878220e-04,}; 
k2c_tensor keras_target_temp_test2 = {&keras_target_temp_test2_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test2_array[33] = {0}; 
k2c_tensor c_target_temp_test2 = {&c_target_temp_test2_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test2_array[33] = {
1.45533100e-01,2.40626961e-01,2.01369882e-01,2.65999973e-01,2.60730565e-01,
3.07292104e-01,4.72149253e-01,3.94623101e-01,4.05062318e-01,3.60198140e-01,
2.87011206e-01,4.43348467e-01,1.98072284e-01,2.31821805e-01,2.32285693e-01,
2.91233901e-02,1.51894346e-01,2.52009451e-01,2.69162655e-01,3.24067533e-01,
2.83387423e-01,2.91346967e-01,2.37235412e-01,5.87911494e-02,-1.50229409e-02,
1.33243352e-01,2.11086199e-02,5.38882390e-02,3.01897451e-02,3.33285928e-02,
3.43625173e-02,2.47173943e-04,1.39528187e-03,}; 
k2c_tensor keras_target_dens_test2 = {&keras_target_dens_test2_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test2_array[33] = {0}; 
k2c_tensor c_target_dens_test2 = {&c_target_dens_test2_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test3_input_thomson_temp_EFITRT1_input_array[33] = {
-9.11140449e-01,-6.48068624e-01,4.41227286e-01,6.54840747e-01,-5.16922992e-01,
-1.78006791e+00,-1.26745569e+00,-1.09695168e+00,4.85380005e-02,-8.27019857e-01,
1.26719585e+00,1.33408554e+00,8.79096828e-01,-1.97875835e-01,1.60024464e+00,
-1.21774689e+00,-8.71201915e-01,-8.92325248e-01,-1.34023457e+00,-1.65439970e+00,
-1.43912828e+00,1.45982961e+00,1.11116334e+00,1.06279944e+00,-5.50438910e-01,
-5.71619109e-01,1.19430824e+00,-1.03884080e+00,-2.27176877e-01,6.60710686e-01,
1.76863403e+00,4.58727526e-01,5.70183421e-01,}; 
k2c_tensor test3_input_thomson_temp_EFITRT1_input = {&test3_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test3_input_thomson_dens_EFITRT1_input_array[33] = {
1.00001829e+00,7.16477960e-01,9.94807985e-01,-7.74196291e-01,-6.29512167e-03,
-1.79713300e-01,2.81562000e-01,-2.68756295e-01,-3.18842130e-01,-6.73467323e-01,
-3.36234302e-01,1.67805866e+00,8.76246852e-01,-1.37731730e+00,1.80821668e+00,
1.02780261e+00,7.16063227e-01,1.55680118e+00,1.33038895e+00,5.04664806e-01,
-9.80873950e-01,5.77073764e-01,7.80863353e-01,-3.48697716e-01,1.59680945e+00,
7.69451559e-01,1.68164157e+00,3.70162569e-01,2.91952293e-01,-1.80587755e+00,
1.71692948e+00,8.80989473e-01,-1.29988476e+00,}; 
k2c_tensor test3_input_thomson_dens_EFITRT1_input = {&test3_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test3_input_past_pinj_input_array[8] = {
-1.39894313e+00,-9.62211624e-01,1.30832658e+00,-1.22775184e+00,1.62261892e-01,
1.92527635e+00,1.16981587e+00,9.08825066e-01,}; 
k2c_tensor test3_input_past_pinj_input = {&test3_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test3_input_past_curr_input_array[8] = {
2.93796721e-01,1.66074697e+00,4.09097927e-02,1.18997595e+00,-1.66543326e+00,
-1.22485294e+00,-1.21254333e+00,1.12837989e+00,}; 
k2c_tensor test3_input_past_curr_input = {&test3_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test3_input_past_tinj_input_array[8] = {
-1.44034352e+00,9.64344484e-01,1.53673505e-01,-1.90453639e+00,-1.18520064e+00,
6.71127838e-02,-7.29160233e-01,-2.11619790e-01,}; 
k2c_tensor test3_input_past_tinj_input = {&test3_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test3_input_past_gasA_input_array[8] = {
2.36816359e-01,-5.90116024e-02,-8.06036334e-01,1.09716729e+00,-1.56878374e+00,
-2.02878381e-01,-7.62717058e-01,-1.07929340e+00,}; 
k2c_tensor test3_input_past_gasA_input = {&test3_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test3_input_future_pinj_input_array[4] = {
-1.86706151e+00,3.66799327e-01,-1.95142131e+00,-1.59179833e+00,}; 
k2c_tensor test3_input_future_pinj_input = {&test3_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test3_input_future_curr_input_array[4] = {
-1.60623827e+00,6.09290763e-01,1.10690986e+00,1.87290048e+00,}; 
k2c_tensor test3_input_future_curr_input = {&test3_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test3_input_future_tinj_input_array[4] = {
-1.03569557e+00,4.80970937e-02,1.73535605e+00,-1.87965623e+00,}; 
k2c_tensor test3_input_future_tinj_input = {&test3_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test3_input_future_gasA_input_array[4] = {
7.47064033e-01,-1.15442872e+00,-1.91847424e+00,1.67055375e+00,}; 
k2c_tensor test3_input_future_gasA_input = {&test3_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test3_array[33] = {
-3.44287306e-02,-1.89502686e-02,-2.13354509e-02,2.51940638e-03,-5.08728530e-03,
2.00120197e-03,-7.61124212e-03,2.69455556e-03,1.09700672e-03,3.54619173e-04,
5.05027128e-04,2.28990975e-05,1.91482919e-04,-1.61200762e-04,-9.54549410e-04,
-9.51147289e-04,1.22824614e-03,-2.05539959e-03,8.56785919e-04,1.78534538e-06,
-4.01536992e-04,-2.99776264e-04,-1.75363850e-04,-4.20913217e-04,-1.78635237e-04,
3.12937162e-04,-2.73670827e-04,-8.52287048e-04,1.22803322e-05,3.14074889e-04,
-1.05061883e-03,2.99064995e-04,7.76721048e-04,}; 
k2c_tensor keras_target_temp_test3 = {&keras_target_temp_test3_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test3_array[33] = {0}; 
k2c_tensor c_target_temp_test3 = {&c_target_temp_test3_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test3_array[33] = {
3.89438987e-01,5.39037645e-01,5.11324883e-01,4.42999184e-01,2.49518454e-01,
4.54208910e-01,6.34354115e-01,4.60627496e-01,2.98292875e-01,1.12055436e-01,
1.06655337e-01,4.30182517e-01,2.33875424e-01,2.32780620e-01,2.43461624e-01,
7.79528171e-03,-1.08956825e-03,-1.31276064e-03,-5.47383260e-03,1.59400925e-02,
-5.07397484e-03,5.45293093e-04,-1.74763054e-03,-8.84502195e-04,-7.42306001e-04,
-5.73221594e-04,-5.72297722e-06,-2.35717744e-06,7.60145485e-05,-7.45220110e-04,
-2.02394929e-03,-2.67134048e-04,-3.32387723e-03,}; 
k2c_tensor keras_target_dens_test3 = {&keras_target_dens_test3_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test3_array[33] = {0}; 
k2c_tensor c_target_dens_test3 = {&c_target_dens_test3_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test4_input_thomson_temp_EFITRT1_input_array[33] = {
1.41860758e+00,1.45761353e+00,1.58060972e+00,-1.24973082e+00,-1.38366407e+00,
7.28291618e-02,5.44719658e-01,-1.48462925e+00,-1.94509687e+00,1.91248758e+00,
3.91878801e-01,1.30656205e+00,-1.60250651e+00,3.57381312e-01,4.09529936e-01,
1.94929598e+00,-3.88707609e-01,-1.73943797e+00,1.28561496e+00,-5.12781959e-01,
1.90359308e+00,1.92113618e-01,-1.19475443e+00,-7.80154925e-01,-1.89384151e-01,
1.33157319e+00,5.12571565e-01,1.13437217e+00,9.88181998e-01,-1.90477050e+00,
-7.22413256e-01,4.19463071e-01,1.99997345e+00,}; 
k2c_tensor test4_input_thomson_temp_EFITRT1_input = {&test4_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test4_input_thomson_dens_EFITRT1_input_array[33] = {
1.00427547e+00,1.62464187e-01,1.44871063e+00,1.53523166e+00,6.11610030e-02,
-1.68474472e+00,-4.65579192e-01,-1.78025542e-01,-2.41866325e-01,4.88639649e-01,
-1.37866539e-01,-1.88616908e+00,1.68497021e+00,-8.41947758e-02,9.65058612e-01,
-1.07928847e+00,-9.00134475e-01,4.35078656e-02,-9.18630914e-02,1.75205751e+00,
5.34274976e-01,1.75788137e+00,5.35669704e-01,1.80702277e-01,3.36553530e-01,
-1.02968229e+00,-1.40345933e+00,6.65872095e-01,2.24586415e-02,-1.77534203e+00,
-9.18932788e-01,1.60180115e+00,1.15648660e+00,}; 
k2c_tensor test4_input_thomson_dens_EFITRT1_input = {&test4_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test4_input_past_pinj_input_array[8] = {
1.55695123e+00,1.37785492e+00,1.48857059e+00,-4.42551094e-01,3.04243609e-01,
2.08343133e-01,-1.31353015e+00,1.20663764e+00,}; 
k2c_tensor test4_input_past_pinj_input = {&test4_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test4_input_past_curr_input_array[8] = {
-7.23769460e-01,-1.42222371e+00,-1.63110216e+00,6.43168759e-01,1.56374185e+00,
-1.32771254e+00,-4.55900714e-01,-4.30367584e-02,}; 
k2c_tensor test4_input_past_curr_input = {&test4_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test4_input_past_tinj_input_array[8] = {
-1.11877983e+00,-1.16305588e+00,5.31980641e-01,-1.39478611e+00,-1.55140746e+00,
1.54045451e+00,-1.16959591e+00,-4.99522313e-01,}; 
k2c_tensor test4_input_past_tinj_input = {&test4_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test4_input_past_gasA_input_array[8] = {
-1.39982115e+00,1.18491423e+00,1.77387423e+00,1.37242127e+00,9.16998755e-01,
-1.76531652e+00,-5.75529023e-01,-1.52938750e+00,}; 
k2c_tensor test4_input_past_gasA_input = {&test4_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test4_input_future_pinj_input_array[4] = {
-1.79640498e+00,-1.40487729e-01,-1.81275835e+00,-6.31537829e-01,}; 
k2c_tensor test4_input_future_pinj_input = {&test4_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test4_input_future_curr_input_array[4] = {
-1.95369139e+00,-1.51080016e+00,1.69795586e+00,9.75602858e-01,}; 
k2c_tensor test4_input_future_curr_input = {&test4_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test4_input_future_tinj_input_array[4] = {
-9.54322667e-01,-7.47322396e-01,9.57136731e-01,-9.92467888e-01,}; 
k2c_tensor test4_input_future_tinj_input = {&test4_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test4_input_future_gasA_input_array[4] = {
8.48890944e-02,3.87002855e-01,6.11521464e-01,-5.67136627e-01,}; 
k2c_tensor test4_input_future_gasA_input = {&test4_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test4_array[33] = {
3.56935110e-04,-1.77390873e-04,4.55297418e-02,4.10823599e-02,4.48626690e-02,
9.95942280e-02,8.75929222e-02,9.90760848e-02,1.29333198e-01,1.54881462e-01,
1.57359794e-01,1.63985386e-01,1.74407169e-01,1.63096413e-01,1.13736644e-01,
4.45121005e-02,2.80436157e-04,9.38022131e-05,6.72171474e-04,-1.21956458e-04,
1.25124148e-04,5.78118255e-04,-1.11193396e-04,1.69076709e-04,1.45621816e-04,
-1.08476530e-03,-2.67792447e-03,2.89524812e-03,-3.53223342e-03,-4.36206907e-03,
-8.02339404e-04,1.21607713e-03,-1.32470299e-03,}; 
k2c_tensor keras_target_temp_test4 = {&keras_target_temp_test4_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test4_array[33] = {0}; 
k2c_tensor c_target_temp_test4 = {&c_target_temp_test4_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test4_array[33] = {
8.71625077e-03,1.83199206e-03,1.53946225e-02,4.28961543e-03,2.02830985e-01,
1.28406137e-01,2.11176500e-01,2.95796037e-01,4.00343776e-01,5.04472494e-01,
4.91751730e-01,3.09401870e-01,3.01449656e-01,3.82431507e-01,3.29107583e-01,
2.09903210e-01,1.03345424e-01,7.98701346e-02,8.63836035e-02,9.35507640e-02,
4.76144627e-03,2.29954487e-03,1.03150355e-03,1.99882314e-04,8.01589340e-04,
-4.09953296e-04,-1.08600128e-03,-9.34102573e-04,-2.75269616e-03,3.00549669e-03,
-2.08749715e-03,1.99767947e-02,-2.46309955e-03,}; 
k2c_tensor keras_target_dens_test4 = {&keras_target_dens_test4_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test4_array[33] = {0}; 
k2c_tensor c_target_dens_test4 = {&c_target_dens_test4_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test5_input_thomson_temp_EFITRT1_input_array[33] = {
2.39379708e-01,1.34603427e+00,8.62092447e-01,-1.61938928e+00,1.65302852e+00,
-9.23421154e-01,-4.63680265e-01,-1.59643809e+00,-1.22024569e+00,-7.11880789e-02,
-1.89986304e+00,1.87108656e+00,4.72130912e-01,-9.50570580e-01,-1.09156330e+00,
-7.22289200e-01,-1.99058082e+00,-1.84552005e+00,2.34827532e-01,-3.75196191e-01,
-1.05319783e+00,-1.06412017e+00,3.60051191e-01,-7.57050902e-01,-1.71291146e+00,
-7.83146704e-01,1.70471694e+00,1.52575819e+00,1.63241644e+00,1.44641935e+00,
-1.21061601e+00,-1.77437615e+00,1.85111464e+00,}; 
k2c_tensor test5_input_thomson_temp_EFITRT1_input = {&test5_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test5_input_thomson_dens_EFITRT1_input_array[33] = {
1.80901963e+00,-9.67757139e-02,2.24154150e-01,-1.03401900e+00,-2.39374418e-01,
6.21132408e-02,-1.99535575e+00,-6.71498674e-01,3.75841486e-01,-4.06286794e-01,
-8.94848043e-02,9.55815646e-01,1.46795930e+00,-1.91097860e+00,-4.18182110e-01,
-6.59251714e-01,8.77443117e-01,-1.65236416e+00,2.00848306e-01,9.41062969e-01,
-1.26104707e+00,-1.19839497e+00,-1.35370357e+00,-1.49704556e+00,-8.80312638e-01,
-6.16525921e-02,1.31017092e+00,1.36329771e+00,-8.82692496e-01,9.96425811e-01,
1.90898434e+00,-1.90670910e-01,1.91896955e+00,}; 
k2c_tensor test5_input_thomson_dens_EFITRT1_input = {&test5_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test5_input_past_pinj_input_array[8] = {
-1.98904381e+00,1.22305769e+00,1.98787670e+00,1.40651933e+00,-1.14611102e-01,
-4.29661447e-01,-1.56186465e+00,-1.73431696e+00,}; 
k2c_tensor test5_input_past_pinj_input = {&test5_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test5_input_past_curr_input_array[8] = {
-7.26429465e-01,-1.34199438e+00,8.51406542e-01,-1.70147730e+00,-5.61238553e-01,
1.12597575e+00,-7.61064979e-02,-1.77436460e+00,}; 
k2c_tensor test5_input_past_curr_input = {&test5_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test5_input_past_tinj_input_array[8] = {
7.12753738e-01,-7.62938537e-01,1.17747266e-01,1.62710762e+00,1.97282832e+00,
-2.42867490e-03,-1.84555697e+00,1.56488216e+00,}; 
k2c_tensor test5_input_past_tinj_input = {&test5_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test5_input_past_gasA_input_array[8] = {
-1.72370673e+00,1.16278581e+00,-1.99519654e+00,-1.08886840e+00,8.71816727e-01,
-1.90664341e+00,1.22729415e+00,-1.03911078e+00,}; 
k2c_tensor test5_input_past_gasA_input = {&test5_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test5_input_future_pinj_input_array[4] = {
-4.09585862e-01,-1.77854321e+00,1.44821629e+00,-2.87447010e-01,}; 
k2c_tensor test5_input_future_pinj_input = {&test5_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test5_input_future_curr_input_array[4] = {
3.71692996e-01,-1.59139901e+00,1.90092837e+00,-1.40830543e-01,}; 
k2c_tensor test5_input_future_curr_input = {&test5_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test5_input_future_tinj_input_array[4] = {
-1.35058937e+00,-6.87122117e-01,-7.73687569e-01,-1.42437017e+00,}; 
k2c_tensor test5_input_future_tinj_input = {&test5_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test5_input_future_gasA_input_array[4] = {
1.89540144e+00,-1.14492963e+00,-1.23959642e+00,7.77451329e-01,}; 
k2c_tensor test5_input_future_gasA_input = {&test5_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test5_array[33] = {
-5.80672931e-05,-2.16072542e-04,-2.52170954e-04,4.38857474e-04,-6.57953147e-04,
-5.19511697e-04,6.28841924e-04,9.00721207e-05,3.54018033e-04,1.88194026e-04,
4.25075705e-05,-4.73540975e-04,-1.18361867e-03,-3.99534823e-04,-1.26281660e-03,
1.18405372e-03,-1.22373700e-02,-1.02863433e-02,-1.00963172e-02,-4.03100736e-02,
-1.16273932e-01,-1.84743121e-01,-3.55542779e-01,-3.25237989e-01,-2.81569988e-01,
-2.41798654e-01,-2.39468217e-01,-2.14876324e-01,-2.10940242e-01,-9.58863571e-02,
3.56011232e-03,1.04630077e-02,-4.21507098e-03,}; 
k2c_tensor keras_target_temp_test5 = {&keras_target_temp_test5_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test5_array[33] = {0}; 
k2c_tensor c_target_temp_test5 = {&c_target_temp_test5_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test5_array[33] = {
2.54430203e-03,3.87393357e-03,2.62761209e-03,2.26836232e-03,4.20570653e-03,
4.79072193e-03,1.33839939e-02,1.90082043e-02,2.46014353e-02,3.25783603e-02,
3.02925110e-02,3.17565575e-02,2.56690606e-02,1.90866981e-02,-1.48639232e-02,
2.02532224e-02,6.16895668e-02,4.19464102e-03,1.88170299e-02,2.14082235e-03,
1.86829641e-03,2.02845130e-03,6.29137130e-03,-4.76368703e-04,1.22199981e-02,
6.59401994e-04,4.80091199e-04,7.58418068e-03,-1.10699032e-02,5.80492895e-03,
2.16576271e-04,2.81891264e-02,-3.32679786e-03,}; 
k2c_tensor keras_target_dens_test5 = {&keras_target_dens_test5_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test5_array[33] = {0}; 
k2c_tensor c_target_dens_test5 = {&c_target_dens_test5_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test6_input_thomson_temp_EFITRT1_input_array[33] = {
1.65534754e+00,-1.94360625e+00,-1.20028935e+00,-1.03017072e-02,-4.18695355e-01,
-6.32175392e-01,-3.50692377e-01,-1.06349751e+00,1.38248518e+00,1.02286944e+00,
1.62847687e+00,-1.52427298e+00,-6.32194955e-01,1.85317818e+00,-5.15663768e-01,
6.48223204e-01,7.91252732e-01,-3.12842915e-01,-6.17109522e-01,-4.57962408e-01,
7.87660040e-01,-1.91411923e+00,-9.54240240e-01,3.54250363e-01,6.95958047e-01,
1.92760794e-01,3.82269802e-01,-5.85843534e-01,-1.37030761e+00,6.37767738e-01,
-4.80971586e-01,-1.66269002e+00,-1.70601591e+00,}; 
k2c_tensor test6_input_thomson_temp_EFITRT1_input = {&test6_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test6_input_thomson_dens_EFITRT1_input_array[33] = {
-5.58577679e-01,3.43207258e-01,4.37619892e-01,-1.54607945e+00,-6.39099057e-01,
1.55829285e+00,1.56139039e+00,1.59244533e+00,1.07261404e+00,1.82681607e+00,
6.96919587e-01,1.62931238e-01,-3.93073613e-01,1.69312922e+00,1.54519334e+00,
-5.62131369e-02,1.33761175e+00,1.82006194e+00,-1.20520440e+00,-4.01498444e-01,
-1.72403035e+00,-1.96222750e+00,-1.02553193e+00,-1.83677111e+00,1.93289477e+00,
4.65457803e-02,9.49382506e-01,-1.97968222e+00,-1.40457612e+00,1.13284319e+00,
-3.67683377e-01,1.28118093e-01,1.52645646e+00,}; 
k2c_tensor test6_input_thomson_dens_EFITRT1_input = {&test6_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test6_input_past_pinj_input_array[8] = {
1.19039884e+00,-4.80022223e-01,-9.04639631e-01,-2.93903560e-01,1.16931126e+00,
8.39690433e-01,-3.76231037e-01,6.01269653e-01,}; 
k2c_tensor test6_input_past_pinj_input = {&test6_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test6_input_past_curr_input_array[8] = {
-1.78852210e-01,-1.80320654e+00,-1.14395124e+00,-1.06028432e+00,1.04097690e+00,
1.39703630e+00,-2.58990670e-01,1.28209688e+00,}; 
k2c_tensor test6_input_past_curr_input = {&test6_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test6_input_past_tinj_input_array[8] = {
-2.01939871e-01,1.75719707e+00,1.61610518e+00,5.87073317e-01,-1.53106901e+00,
-2.16455197e-01,-3.72842954e-01,1.91634820e+00,}; 
k2c_tensor test6_input_past_tinj_input = {&test6_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test6_input_past_gasA_input_array[8] = {
-1.93668845e+00,7.26541822e-02,-1.90504211e+00,-1.23678628e+00,1.37103468e+00,
5.06275242e-01,1.38737394e+00,-1.14469719e+00,}; 
k2c_tensor test6_input_past_gasA_input = {&test6_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test6_input_future_pinj_input_array[4] = {
1.78084125e+00,1.21182901e+00,-5.07598820e-01,1.93061137e+00,}; 
k2c_tensor test6_input_future_pinj_input = {&test6_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test6_input_future_curr_input_array[4] = {
-7.03708823e-01,1.78879861e+00,1.94095274e+00,8.12433539e-01,}; 
k2c_tensor test6_input_future_curr_input = {&test6_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test6_input_future_tinj_input_array[4] = {
1.00923196e+00,4.74702093e-01,-7.76692491e-01,-7.70659423e-01,}; 
k2c_tensor test6_input_future_tinj_input = {&test6_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test6_input_future_gasA_input_array[4] = {
-1.43065260e+00,-1.72291150e+00,-1.68063896e+00,1.43840368e+00,}; 
k2c_tensor test6_input_future_gasA_input = {&test6_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test6_array[33] = {
4.39948984e-04,3.70534632e-04,-3.93561262e-04,7.91135477e-04,1.17599513e-04,
-1.02263177e-04,4.92441817e-04,-1.93957763e-04,-3.38831625e-04,1.42842662e-04,
-1.25982845e-03,-2.59428867e-04,4.60232550e-04,4.80277911e-02,4.78772782e-02,
5.91048934e-02,7.58772045e-02,7.02060610e-02,6.38668984e-02,5.07649444e-02,
1.86959188e-03,-9.02419328e-04,-9.23720072e-05,2.33234558e-03,-1.14550942e-03,
5.41257858e-03,2.12113783e-02,1.84824374e-02,3.56371631e-03,4.77409735e-03,
1.18227955e-03,6.19197451e-03,3.49176698e-03,}; 
k2c_tensor keras_target_temp_test6 = {&keras_target_temp_test6_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test6_array[33] = {0}; 
k2c_tensor c_target_temp_test6 = {&c_target_temp_test6_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test6_array[33] = {
3.53881856e-03,6.40719850e-03,7.10197724e-03,7.76418718e-03,7.40786269e-03,
6.83491956e-03,2.51943106e-03,2.05698004e-03,4.44012694e-04,2.76292674e-04,
-1.13936421e-03,1.35109853e-03,1.56967202e-03,4.25842963e-03,5.81494905e-03,
7.06329755e-03,8.99356324e-03,7.17455475e-03,3.82536044e-03,-1.28443073e-03,
6.90285256e-03,2.47859908e-03,3.37683596e-04,2.55486369e-03,1.62035041e-03,
3.08836019e-03,4.93235607e-03,6.06728857e-03,1.60347717e-03,3.44736315e-03,
-2.21563969e-03,4.66263015e-03,9.45429597e-03,}; 
k2c_tensor keras_target_dens_test6 = {&keras_target_dens_test6_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test6_array[33] = {0}; 
k2c_tensor c_target_dens_test6 = {&c_target_dens_test6_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test7_input_thomson_temp_EFITRT1_input_array[33] = {
-1.02373567e+00,2.35308551e-01,-8.79747453e-01,1.10563154e+00,1.17014229e+00,
-5.19150311e-01,5.33541070e-02,-4.50919277e-01,1.85726639e+00,-1.02851714e-01,
9.39357241e-01,3.91213848e-01,1.65825029e+00,4.72465839e-01,-1.38998447e+00,
-1.89840148e+00,-6.70045297e-01,-8.39992735e-01,8.37680902e-01,-1.96321039e+00,
7.95258336e-01,-1.28501628e+00,-1.85580984e+00,1.18066349e+00,-3.23332891e-01,
2.56917128e-01,4.07189803e-01,-9.34307735e-01,1.63104403e+00,3.68541095e-01,
6.87748904e-01,1.92155710e+00,-5.87520243e-01,}; 
k2c_tensor test7_input_thomson_temp_EFITRT1_input = {&test7_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test7_input_thomson_dens_EFITRT1_input_array[33] = {
1.66549118e+00,5.93448059e-01,1.82669877e+00,-1.73393571e+00,-8.47948657e-01,
-1.12158642e-01,-1.45298937e+00,1.53546824e+00,8.38958963e-01,-5.97458108e-01,
-1.24806153e+00,-5.55489958e-01,-1.97388212e+00,1.34645797e+00,-2.82602545e-01,
-1.18398271e+00,-1.07217655e+00,-1.58758277e+00,-2.34435070e-01,7.61472396e-01,
-5.65485491e-01,1.25825907e+00,-1.44504429e+00,8.31375463e-01,-3.21868219e-02,
1.44712161e+00,-7.86455212e-01,-1.07984635e+00,8.55552625e-02,1.55704069e+00,
1.15460492e-01,-6.52892846e-01,1.72093175e+00,}; 
k2c_tensor test7_input_thomson_dens_EFITRT1_input = {&test7_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test7_input_past_pinj_input_array[8] = {
1.49086147e+00,-1.44622459e+00,8.46324776e-01,-2.94777032e-01,-6.44394848e-02,
1.56714748e-01,-1.27072023e+00,-1.22958724e+00,}; 
k2c_tensor test7_input_past_pinj_input = {&test7_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test7_input_past_curr_input_array[8] = {
2.46771710e-01,-9.97845304e-01,-1.47499787e+00,4.60742263e-01,-1.66849994e+00,
1.20614777e+00,7.62678944e-01,2.34316590e-02,}; 
k2c_tensor test7_input_past_curr_input = {&test7_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test7_input_past_tinj_input_array[8] = {
-1.65100220e+00,1.55047592e+00,-1.11880154e+00,-2.77750655e-01,7.36475053e-01,
-1.86325978e+00,-1.01437490e+00,-1.50134356e-01,}; 
k2c_tensor test7_input_past_tinj_input = {&test7_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test7_input_past_gasA_input_array[8] = {
7.91710486e-01,-3.88123429e-01,5.60935676e-01,-7.88003046e-01,8.86954656e-01,
2.64296200e-01,-1.38019989e+00,-6.25670862e-01,}; 
k2c_tensor test7_input_past_gasA_input = {&test7_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test7_input_future_pinj_input_array[4] = {
7.71962574e-01,1.13690084e+00,-1.07144034e+00,-1.59007019e+00,}; 
k2c_tensor test7_input_future_pinj_input = {&test7_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test7_input_future_curr_input_array[4] = {
-1.53864132e+00,-1.92902657e+00,1.30095855e+00,-4.14061978e-01,}; 
k2c_tensor test7_input_future_curr_input = {&test7_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test7_input_future_tinj_input_array[4] = {
6.31562968e-01,2.05198567e-01,-5.70561361e-01,3.83093739e-01,}; 
k2c_tensor test7_input_future_tinj_input = {&test7_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test7_input_future_gasA_input_array[4] = {
1.13004635e+00,-5.97155218e-01,6.96143280e-01,-1.73185782e+00,}; 
k2c_tensor test7_input_future_gasA_input = {&test7_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test7_array[33] = {
9.83559936e-02,1.87087476e-01,1.58941269e-01,1.22818552e-01,1.01106599e-01,
7.59681016e-02,4.92360741e-02,2.73966901e-02,-1.66687532e-05,1.62851589e-04,
-1.73450069e-04,1.61643082e-03,1.45722483e-03,1.18104242e-01,1.77023038e-01,
3.09139371e-01,4.11057413e-01,5.41283488e-01,5.33831298e-01,5.31718194e-01,
5.26391923e-01,4.32651937e-01,4.28495049e-01,3.68956238e-01,2.63764769e-01,
2.02051118e-01,1.50309756e-01,7.47529343e-02,1.80324714e-05,1.61318909e-04,
-2.46889598e-04,1.09363243e-03,2.97415582e-03,}; 
k2c_tensor keras_target_temp_test7 = {&keras_target_temp_test7_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test7_array[33] = {0}; 
k2c_tensor c_target_temp_test7 = {&c_target_temp_test7_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test7_array[33] = {
-7.98171386e-04,-3.15790065e-04,-3.92539427e-04,-2.42236070e-04,-5.05875796e-04,
1.72139378e-03,2.54415767e-03,6.69783279e-02,6.50221407e-02,8.18440244e-02,
1.48590550e-01,3.33825886e-01,4.35705721e-01,4.85065281e-01,4.63927031e-01,
5.35071850e-01,5.80256939e-01,6.86909258e-01,7.24804044e-01,5.68688273e-01,
8.11306417e-01,2.69381106e-01,1.42915010e-01,1.62244454e-01,1.58123493e-01,
5.01408428e-03,3.10449535e-03,9.72245820e-04,-1.07478350e-04,-2.52985395e-04,
-1.58342160e-03,1.33809466e-02,2.28871265e-03,}; 
k2c_tensor keras_target_dens_test7 = {&keras_target_dens_test7_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test7_array[33] = {0}; 
k2c_tensor c_target_dens_test7 = {&c_target_dens_test7_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test8_input_thomson_temp_EFITRT1_input_array[33] = {
1.75542942e+00,-1.39329423e+00,-1.54477396e+00,1.54031699e-01,-9.44442046e-01,
1.56579432e+00,1.26355040e+00,5.97341574e-01,-4.64927032e-02,8.41660135e-01,
1.08306846e+00,-5.84150552e-01,-8.92126879e-01,-9.83027206e-01,-1.96306796e+00,
-7.18357086e-02,2.36885792e-01,-1.37544880e+00,-3.55092661e-01,-1.71108897e+00,
1.45736741e+00,-1.97291246e+00,5.13000513e-01,1.46039800e+00,1.39707191e+00,
-2.46316014e-01,1.86921914e+00,1.54687026e+00,1.10712436e+00,-1.56593618e-01,
1.96526210e+00,-5.42775612e-01,3.71701108e-01,}; 
k2c_tensor test8_input_thomson_temp_EFITRT1_input = {&test8_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test8_input_thomson_dens_EFITRT1_input_array[33] = {
-1.43362311e-01,-5.61367871e-01,1.37143987e+00,-2.98304825e-01,-2.71703765e-01,
-7.32355639e-02,-1.84409372e+00,-1.14775338e+00,-1.65626038e+00,1.64303253e+00,
6.87436334e-01,1.30958077e+00,1.39582023e+00,-2.91236722e-01,-7.58093200e-02,
1.36357120e+00,1.84709435e+00,-1.91178070e+00,-9.84296349e-01,6.70054905e-01,
-1.90991889e+00,1.30560547e+00,1.75796220e+00,1.30378052e+00,-8.98968072e-01,
1.41048963e+00,-7.82767045e-01,-7.23719752e-01,1.21328939e+00,-1.91132284e+00,
-6.08908296e-02,1.55822942e+00,5.10665066e-01,}; 
k2c_tensor test8_input_thomson_dens_EFITRT1_input = {&test8_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test8_input_past_pinj_input_array[8] = {
-1.69353974e+00,1.22017261e+00,1.50499055e+00,8.81291873e-02,1.87034377e+00,
9.18748007e-01,-7.51292943e-01,4.12296249e-01,}; 
k2c_tensor test8_input_past_pinj_input = {&test8_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test8_input_past_curr_input_array[8] = {
-1.44258409e+00,-1.73353047e+00,-1.67081186e+00,-1.75950588e+00,4.50038373e-01,
-1.35729450e+00,-1.34346676e+00,9.15371659e-01,}; 
k2c_tensor test8_input_past_curr_input = {&test8_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test8_input_past_tinj_input_array[8] = {
1.23960506e+00,-1.23265507e+00,-1.92434479e+00,-1.72896005e-01,1.72959140e+00,
-1.10179320e+00,-1.54335718e+00,9.88086896e-01,}; 
k2c_tensor test8_input_past_tinj_input = {&test8_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test8_input_past_gasA_input_array[8] = {
-4.98745062e-01,-6.47936956e-01,8.21505007e-01,1.10610456e+00,-3.29103979e-01,
-1.98320283e+00,-1.93125503e+00,1.75410488e+00,}; 
k2c_tensor test8_input_past_gasA_input = {&test8_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test8_input_future_pinj_input_array[4] = {
1.38646242e+00,-1.89790265e+00,-1.80502334e+00,1.88900678e+00,}; 
k2c_tensor test8_input_future_pinj_input = {&test8_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test8_input_future_curr_input_array[4] = {
7.77625303e-01,-5.50566123e-01,-1.76308160e-01,-1.05875040e+00,}; 
k2c_tensor test8_input_future_curr_input = {&test8_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test8_input_future_tinj_input_array[4] = {
1.90218584e-01,7.00652999e-01,1.30151662e+00,6.23088536e-01,}; 
k2c_tensor test8_input_future_tinj_input = {&test8_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test8_input_future_gasA_input_array[4] = {
1.69843408e+00,7.10720058e-01,1.10998676e+00,4.97321673e-01,}; 
k2c_tensor test8_input_future_gasA_input = {&test8_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test8_array[33] = {
-9.48665664e-04,-2.73867254e-03,-2.69367732e-02,-1.47191048e-01,-1.36206493e-01,
-2.45076478e-01,-2.10478693e-01,-1.97736815e-01,-2.28894487e-01,-2.12984309e-01,
-1.58638775e-01,-1.62011012e-01,7.82682188e-03,1.13740645e-01,9.75793600e-02,
3.53698991e-02,4.41315118e-03,-6.54471864e-04,-2.47507356e-03,-3.99274053e-04,
-6.20807114e-04,-1.39525859e-03,1.05184692e-04,-4.26674320e-04,-2.93027959e-04,
-2.95171980e-04,-2.36123742e-04,-4.88929800e-06,-5.52658748e-05,1.78268238e-04,
1.21715071e-04,1.01633091e-03,-8.66910443e-04,}; 
k2c_tensor keras_target_temp_test8 = {&keras_target_temp_test8_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test8_array[33] = {0}; 
k2c_tensor c_target_temp_test8 = {&c_target_temp_test8_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test8_array[33] = {
1.42041054e-02,-2.39373650e-03,-9.52972099e-04,-1.15764327e-03,-2.42095441e-03,
-5.89076430e-04,-7.11690411e-02,-7.99220130e-02,-2.13911891e-01,-2.03002483e-01,
-2.52425969e-01,-2.39255026e-01,-3.07985663e-01,-3.14403713e-01,-3.59251797e-01,
-2.03727156e-01,-6.83732256e-02,-1.95971839e-02,-2.04153672e-01,-1.82099015e-01,
-1.74251527e-01,-1.92712232e-01,-1.74273923e-01,-1.83679610e-01,-1.83608234e-01,
-7.51781613e-02,6.27832487e-05,6.96370378e-04,-1.39586627e-05,-6.11728057e-04,
-2.98215263e-03,3.41321016e-03,-7.19363801e-04,}; 
k2c_tensor keras_target_dens_test8 = {&keras_target_dens_test8_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test8_array[33] = {0}; 
k2c_tensor c_target_dens_test8 = {&c_target_dens_test8_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test9_input_thomson_temp_EFITRT1_input_array[33] = {
-5.17457219e-01,-9.18331811e-01,-1.85821401e-01,-3.32337654e-01,8.65334980e-01,
5.32430125e-01,3.13038826e-01,1.11911678e+00,-1.73188766e+00,5.94288474e-02,
1.09628398e+00,4.30289767e-01,-7.42915057e-01,2.35853368e-01,7.73194520e-01,
1.68047076e+00,2.50721778e-01,6.70484968e-01,1.03974862e+00,9.61729218e-01,
1.95586649e+00,-1.11459665e+00,1.75478210e+00,-3.44868485e-01,-4.08897189e-01,
-1.21664515e+00,1.62410106e+00,1.85739753e+00,9.66283173e-01,1.41242844e+00,
-4.63462427e-01,1.86649014e+00,9.78317483e-01,}; 
k2c_tensor test9_input_thomson_temp_EFITRT1_input = {&test9_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test9_input_thomson_dens_EFITRT1_input_array[33] = {
4.57189756e-01,-1.02862488e+00,-1.46053608e+00,2.84680372e-01,1.27982237e+00,
3.86167407e-01,-3.99444566e-03,-7.22730851e-01,1.12495908e+00,9.58004469e-02,
-4.00536694e-03,1.12518229e+00,1.53490270e+00,-1.06388400e+00,-1.94876421e+00,
-9.83274977e-01,1.61754092e+00,-4.66694405e-01,1.83177349e+00,-1.26409211e+00,
-1.03946172e+00,-1.90391199e+00,-7.64399680e-01,5.71734362e-01,5.94812376e-01,
3.43095159e-01,-6.42493357e-01,-4.95947690e-01,3.34069783e-01,-5.04411935e-02,
8.14048478e-01,6.66672422e-01,1.75193069e+00,}; 
k2c_tensor test9_input_thomson_dens_EFITRT1_input = {&test9_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test9_input_past_pinj_input_array[8] = {
-1.38726709e+00,4.32983631e-01,-3.90420476e-01,1.60416219e+00,-1.98639476e+00,
-1.28685010e+00,1.95112214e+00,-1.34184418e+00,}; 
k2c_tensor test9_input_past_pinj_input = {&test9_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test9_input_past_curr_input_array[8] = {
-1.12172500e+00,-3.78346327e-01,-1.76961591e+00,-1.64435818e+00,-8.88523609e-01,
8.81566427e-01,-1.29993436e+00,-4.17865932e-01,}; 
k2c_tensor test9_input_past_curr_input = {&test9_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test9_input_past_tinj_input_array[8] = {
1.29465904e+00,-7.78906363e-01,-1.60996724e+00,1.33015908e+00,9.67186835e-01,
3.73991375e-01,1.52781022e+00,1.58902340e-01,}; 
k2c_tensor test9_input_past_tinj_input = {&test9_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test9_input_past_gasA_input_array[8] = {
1.31450177e+00,-1.51948132e-01,-5.24979016e-01,1.24209794e+00,-1.28269341e+00,
6.67594864e-01,1.25666175e+00,1.81338638e+00,}; 
k2c_tensor test9_input_past_gasA_input = {&test9_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test9_input_future_pinj_input_array[4] = {
-9.35322592e-01,-1.47679832e-01,1.39342120e+00,1.37189945e+00,}; 
k2c_tensor test9_input_future_pinj_input = {&test9_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test9_input_future_curr_input_array[4] = {
1.57086137e+00,-9.50476161e-01,-5.66259880e-01,-3.79897625e-01,}; 
k2c_tensor test9_input_future_curr_input = {&test9_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test9_input_future_tinj_input_array[4] = {
-3.53258639e-02,-9.34923853e-02,-6.54271113e-01,-1.00462941e+00,}; 
k2c_tensor test9_input_future_tinj_input = {&test9_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test9_input_future_gasA_input_array[4] = {
7.80071838e-03,-1.90398968e+00,-5.42856135e-01,7.02316377e-01,}; 
k2c_tensor test9_input_future_gasA_input = {&test9_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test9_array[33] = {
-2.61181383e-04,-4.01592522e-04,-1.75273657e-04,-3.18484171e-03,-7.87375122e-02,
-9.82543901e-02,-9.70983431e-02,-8.73367786e-02,-5.96101545e-02,-1.09113693e-01,
-1.10658936e-01,-1.00098714e-01,-1.46259755e-01,-1.32512182e-01,-2.18226641e-01,
-2.25405797e-01,-2.17445850e-01,-2.11616874e-01,-2.72589803e-01,-3.96349609e-01,
-5.34094751e-01,-6.26514912e-01,-6.95309281e-01,-6.85846865e-01,-6.70905948e-01,
-7.59260118e-01,-7.08825290e-01,-6.70539260e-01,-5.87482393e-01,-4.78511482e-01,
-4.27050292e-01,-3.55502635e-01,-2.44087130e-01,}; 
k2c_tensor keras_target_temp_test9 = {&keras_target_temp_test9_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test9_array[33] = {0}; 
k2c_tensor c_target_temp_test9 = {&c_target_temp_test9_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test9_array[33] = {
3.83004732e-03,-1.80356205e-04,2.09768303e-04,1.94869004e-04,2.73680314e-04,
8.97260848e-04,6.74602110e-04,-4.95077111e-04,1.50096603e-04,1.66601362e-03,
1.18066557e-04,-2.10332312e-03,1.90587947e-03,2.71843746e-05,7.06819352e-04,
2.48415396e-04,2.86908355e-03,-2.26806849e-04,7.27069890e-03,3.92778311e-03,
1.72190461e-03,1.04470551e-02,-1.12186354e-02,-3.05837356e-02,-1.65856741e-02,
-3.44707631e-02,-3.38689052e-02,-2.43428349e-02,-3.32197659e-02,-4.08948064e-02,
-1.20382290e-03,2.19382197e-02,1.59657188e-03,}; 
k2c_tensor keras_target_dens_test9 = {&keras_target_dens_test9_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test9_array[33] = {0}; 
k2c_tensor c_target_dens_test9 = {&c_target_dens_test9_array[0],1,33,{33, 1, 1, 1, 1}}; 
float test10_input_thomson_temp_EFITRT1_input_array[33] = {
4.79316658e-01,4.03026761e-01,9.38976149e-01,-7.61295725e-01,8.20119249e-01,
-1.37834853e+00,-1.50707892e+00,-3.44807797e-01,1.16373338e+00,-8.65038608e-01,
-1.11417977e+00,-1.78235391e+00,-1.92584735e+00,-7.56587392e-02,1.69657897e+00,
1.37573046e+00,8.69656583e-01,-1.74882229e-01,1.05290521e+00,9.21792155e-01,
6.79570680e-01,1.39388492e+00,-6.95460207e-01,5.90182157e-01,1.32691429e-01,
-1.84961893e+00,-6.06177161e-01,2.77919495e-01,4.37058445e-01,-1.36768309e+00,
1.64511923e+00,-1.87786668e+00,7.46895135e-01,}; 
k2c_tensor test10_input_thomson_temp_EFITRT1_input = {&test10_input_thomson_temp_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test10_input_thomson_dens_EFITRT1_input_array[33] = {
1.75533561e+00,1.67905840e+00,-1.06230617e+00,-1.98003925e+00,-1.71828458e+00,
-1.67432944e+00,7.82083025e-01,-1.91854957e+00,-3.37729006e-01,3.36663103e-01,
1.62345608e+00,-1.13502856e+00,4.91246529e-01,1.71332815e+00,-1.83920136e+00,
7.27480983e-01,-1.08655617e+00,1.35358208e+00,-8.44283900e-01,5.16347113e-01,
1.89581119e+00,1.75396148e+00,1.17554020e+00,-6.72125403e-01,1.53985158e+00,
6.33402672e-01,-8.42920846e-01,1.32525258e+00,3.85702854e-01,-1.85484760e+00,
-1.76941949e+00,1.43443625e+00,-5.40525529e-01,}; 
k2c_tensor test10_input_thomson_dens_EFITRT1_input = {&test10_input_thomson_dens_EFITRT1_input_array[0],2,33,{ 1,33, 1, 1, 1}}; 
float test10_input_past_pinj_input_array[8] = {
-5.32192509e-01,-2.75766806e-01,1.25032873e+00,8.59130148e-01,1.28249843e+00,
2.30819640e-01,2.89338415e-03,1.72659315e+00,}; 
k2c_tensor test10_input_past_pinj_input = {&test10_input_past_pinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test10_input_past_curr_input_array[8] = {
5.46371363e-01,-2.86994319e-01,-8.15916626e-01,-6.10801004e-01,1.28103650e+00,
1.65333890e+00,-1.86272962e-01,-1.10109346e+00,}; 
k2c_tensor test10_input_past_curr_input = {&test10_input_past_curr_input_array[0],1,8,{8,1,1,1,1}}; 
float test10_input_past_tinj_input_array[8] = {
9.76661213e-01,-7.41817751e-01,1.91707380e+00,1.63347589e+00,-1.07914578e+00,
1.81352520e+00,-1.70840259e+00,6.93347161e-01,}; 
k2c_tensor test10_input_past_tinj_input = {&test10_input_past_tinj_input_array[0],1,8,{8,1,1,1,1}}; 
float test10_input_past_gasA_input_array[8] = {
-1.20589039e+00,2.39059159e-01,-1.97773515e+00,-1.71962514e+00,-1.45798549e+00,
-1.84104542e+00,-1.10862066e+00,1.64736064e+00,}; 
k2c_tensor test10_input_past_gasA_input = {&test10_input_past_gasA_input_array[0],1,8,{8,1,1,1,1}}; 
float test10_input_future_pinj_input_array[4] = {
6.05950895e-01,-8.58230003e-01,6.99072314e-01,-9.23641076e-01,}; 
k2c_tensor test10_input_future_pinj_input = {&test10_input_future_pinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test10_input_future_curr_input_array[4] = {
5.37335050e-01,-9.94417973e-01,6.55696614e-01,-1.86789161e-01,}; 
k2c_tensor test10_input_future_curr_input = {&test10_input_future_curr_input_array[0],1,4,{4,1,1,1,1}}; 
float test10_input_future_tinj_input_array[4] = {
1.95185815e+00,-2.48600213e-01,6.58374657e-01,5.89531639e-01,}; 
k2c_tensor test10_input_future_tinj_input = {&test10_input_future_tinj_input_array[0],1,4,{4,1,1,1,1}}; 
float test10_input_future_gasA_input_array[4] = {
-1.77628053e+00,-9.55242466e-01,9.54374257e-01,-1.23137461e+00,}; 
k2c_tensor test10_input_future_gasA_input = {&test10_input_future_gasA_input_array[0],1,4,{4,1,1,1,1}}; 
float keras_target_temp_test10_array[33] = {
-7.68862478e-03,-7.98141956e-03,-1.41546000e-02,-4.19617107e-04,1.08947419e-03,
-2.48272764e-03,-4.41117305e-03,-1.93005195e-03,-4.91366349e-03,1.06920507e-02,
2.71338727e-02,2.26946492e-02,-2.97365501e-03,-1.71691731e-01,-2.16056556e-01,
-2.06139266e-01,-1.99214339e-01,-1.56823784e-01,-1.90919787e-01,-2.79278815e-01,
-2.13358819e-01,-1.55689120e-01,-1.18628569e-01,-5.58704250e-02,4.74490877e-03,
6.15467317e-03,1.50420638e-02,3.88174574e-03,-5.34560997e-03,-5.29480539e-03,
-5.66812325e-03,-1.05535309e-03,-1.19175587e-03,}; 
k2c_tensor keras_target_temp_test10 = {&keras_target_temp_test10_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_temp_test10_array[33] = {0}; 
k2c_tensor c_target_temp_test10 = {&c_target_temp_test10_array[0],1,33,{33, 1, 1, 1, 1}}; 
float keras_target_dens_test10_array[33] = {
5.51950699e-03,1.05236564e-03,1.30634699e-02,1.06262323e-02,1.25895673e-02,
3.78396958e-02,-4.63947281e-03,7.53551954e-03,2.47380789e-03,1.16207916e-03,
4.37125657e-03,2.14819657e-03,-4.82849032e-02,-1.11593440e-01,-1.04325898e-01,
-1.36863157e-01,-1.30010426e-01,-9.63499472e-02,-7.72537664e-02,-6.37423843e-02,
-2.15417221e-02,-2.75576375e-02,-1.86064653e-03,-2.23497618e-02,-3.24919000e-02,
-2.45984569e-02,-3.93428169e-02,-2.16975473e-02,-1.27769057e-02,-1.39450524e-02,
-2.13430449e-03,-1.93032529e-03,-2.53840350e-03,}; 
k2c_tensor keras_target_dens_test10 = {&keras_target_dens_test10_array[0],1,33,{33, 1, 1, 1, 1}}; 
float c_target_dens_test10_array[33] = {0}; 
k2c_tensor c_target_dens_test10 = {&c_target_dens_test10_array[0],1,33,{33, 1, 1, 1, 1}}; 
 float errors[20];
 size_t num_tests = 10; 
size_t num_outputs = 2; 
etemp_profile_predictor_initialize(); 
clock_t t0 = clock(); 
etemp_profile_predictor(&test1_input_thomson_temp_EFITRT1_input,&test1_input_thomson_dens_EFITRT1_input,&test1_input_past_pinj_input,&test1_input_past_curr_input,&test1_input_past_tinj_input,&test1_input_past_gasA_input,&test1_input_future_pinj_input,&test1_input_future_curr_input,&test1_input_future_tinj_input,&test1_input_future_gasA_input,&c_target_temp_test1,&c_target_dens_test1); 
etemp_profile_predictor(&test2_input_thomson_temp_EFITRT1_input,&test2_input_thomson_dens_EFITRT1_input,&test2_input_past_pinj_input,&test2_input_past_curr_input,&test2_input_past_tinj_input,&test2_input_past_gasA_input,&test2_input_future_pinj_input,&test2_input_future_curr_input,&test2_input_future_tinj_input,&test2_input_future_gasA_input,&c_target_temp_test2,&c_target_dens_test2); 
etemp_profile_predictor(&test3_input_thomson_temp_EFITRT1_input,&test3_input_thomson_dens_EFITRT1_input,&test3_input_past_pinj_input,&test3_input_past_curr_input,&test3_input_past_tinj_input,&test3_input_past_gasA_input,&test3_input_future_pinj_input,&test3_input_future_curr_input,&test3_input_future_tinj_input,&test3_input_future_gasA_input,&c_target_temp_test3,&c_target_dens_test3); 
etemp_profile_predictor(&test4_input_thomson_temp_EFITRT1_input,&test4_input_thomson_dens_EFITRT1_input,&test4_input_past_pinj_input,&test4_input_past_curr_input,&test4_input_past_tinj_input,&test4_input_past_gasA_input,&test4_input_future_pinj_input,&test4_input_future_curr_input,&test4_input_future_tinj_input,&test4_input_future_gasA_input,&c_target_temp_test4,&c_target_dens_test4); 
etemp_profile_predictor(&test5_input_thomson_temp_EFITRT1_input,&test5_input_thomson_dens_EFITRT1_input,&test5_input_past_pinj_input,&test5_input_past_curr_input,&test5_input_past_tinj_input,&test5_input_past_gasA_input,&test5_input_future_pinj_input,&test5_input_future_curr_input,&test5_input_future_tinj_input,&test5_input_future_gasA_input,&c_target_temp_test5,&c_target_dens_test5); 
etemp_profile_predictor(&test6_input_thomson_temp_EFITRT1_input,&test6_input_thomson_dens_EFITRT1_input,&test6_input_past_pinj_input,&test6_input_past_curr_input,&test6_input_past_tinj_input,&test6_input_past_gasA_input,&test6_input_future_pinj_input,&test6_input_future_curr_input,&test6_input_future_tinj_input,&test6_input_future_gasA_input,&c_target_temp_test6,&c_target_dens_test6); 
etemp_profile_predictor(&test7_input_thomson_temp_EFITRT1_input,&test7_input_thomson_dens_EFITRT1_input,&test7_input_past_pinj_input,&test7_input_past_curr_input,&test7_input_past_tinj_input,&test7_input_past_gasA_input,&test7_input_future_pinj_input,&test7_input_future_curr_input,&test7_input_future_tinj_input,&test7_input_future_gasA_input,&c_target_temp_test7,&c_target_dens_test7); 
etemp_profile_predictor(&test8_input_thomson_temp_EFITRT1_input,&test8_input_thomson_dens_EFITRT1_input,&test8_input_past_pinj_input,&test8_input_past_curr_input,&test8_input_past_tinj_input,&test8_input_past_gasA_input,&test8_input_future_pinj_input,&test8_input_future_curr_input,&test8_input_future_tinj_input,&test8_input_future_gasA_input,&c_target_temp_test8,&c_target_dens_test8); 
etemp_profile_predictor(&test9_input_thomson_temp_EFITRT1_input,&test9_input_thomson_dens_EFITRT1_input,&test9_input_past_pinj_input,&test9_input_past_curr_input,&test9_input_past_tinj_input,&test9_input_past_gasA_input,&test9_input_future_pinj_input,&test9_input_future_curr_input,&test9_input_future_tinj_input,&test9_input_future_gasA_input,&c_target_temp_test9,&c_target_dens_test9); 
etemp_profile_predictor(&test10_input_thomson_temp_EFITRT1_input,&test10_input_thomson_dens_EFITRT1_input,&test10_input_past_pinj_input,&test10_input_past_curr_input,&test10_input_past_tinj_input,&test10_input_past_gasA_input,&test10_input_future_pinj_input,&test10_input_future_curr_input,&test10_input_future_tinj_input,&test10_input_future_gasA_input,&c_target_temp_test10,&c_target_dens_test10); 

clock_t t1 = clock(); 
printf("Average time over 10 tests: %e s \n",(double)(t1-t0)/(double)CLOCKS_PER_SEC/(double)10); 
errors[0] = maxabs(&keras_target_temp_test1,&c_target_temp_test1); 
errors[1] = maxabs(&keras_target_dens_test1,&c_target_dens_test1); 
errors[2] = maxabs(&keras_target_temp_test2,&c_target_temp_test2); 
errors[3] = maxabs(&keras_target_dens_test2,&c_target_dens_test2); 
errors[4] = maxabs(&keras_target_temp_test3,&c_target_temp_test3); 
errors[5] = maxabs(&keras_target_dens_test3,&c_target_dens_test3); 
errors[6] = maxabs(&keras_target_temp_test4,&c_target_temp_test4); 
errors[7] = maxabs(&keras_target_dens_test4,&c_target_dens_test4); 
errors[8] = maxabs(&keras_target_temp_test5,&c_target_temp_test5); 
errors[9] = maxabs(&keras_target_dens_test5,&c_target_dens_test5); 
errors[10] = maxabs(&keras_target_temp_test6,&c_target_temp_test6); 
errors[11] = maxabs(&keras_target_dens_test6,&c_target_dens_test6); 
errors[12] = maxabs(&keras_target_temp_test7,&c_target_temp_test7); 
errors[13] = maxabs(&keras_target_dens_test7,&c_target_dens_test7); 
errors[14] = maxabs(&keras_target_temp_test8,&c_target_temp_test8); 
errors[15] = maxabs(&keras_target_dens_test8,&c_target_dens_test8); 
errors[16] = maxabs(&keras_target_temp_test9,&c_target_temp_test9); 
errors[17] = maxabs(&keras_target_dens_test9,&c_target_dens_test9); 
errors[18] = maxabs(&keras_target_temp_test10,&c_target_temp_test10); 
errors[19] = maxabs(&keras_target_dens_test10,&c_target_dens_test10); 
float maxerror = errors[0]; 
for(size_t i=1; i< num_tests*num_outputs;i++){ 
if (errors[i] > maxerror) { 
maxerror = errors[i];}} 
printf("Max absolute error for 10 tests: %e \n", maxerror);
etemp_profile_predictor_terminate(); 
if (maxerror > 1e-05) { 
return 1;} 
return 0;
} 

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2){ 

    float x = 0; 

    float y = 0; 

    for(size_t i=0; i<tensor1->numel; i++){

    y = fabs(tensor1->array[i]-tensor2->array[i]);
    if (y>x) {x=y;}}
    return x;}

