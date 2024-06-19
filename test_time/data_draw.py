import matplotlib.pyplot as plt
import numpy as np
import sys

root_path="/dynamic_batch/triton-multi-modal-serving"

values1=iopen=[0.03638505935668945,0.284343957901001,0.570300817489624,0.8558990955352783,1.1355152130126953]
image_preprocess=[0.011359930038452148,0.08518099784851074,0.16853594779968262,0.2539658546447754,0.3367917537689209]
values2=i_i=[iopen[i]+image_preprocess[i] for i in range(len(iopen))] 
visual_encoder=[ 0.017341567993164064,0.11793119812011718,0.23538368225097656,0.3516784362792969, 0.479321044921875]
values3=i_i_v=[i_i[i]+visual_encoder[i] for i in range(len(i_i))]
values4=text_preprocess=[ 0.0010776640176773072,0.004596704006195068,0.008595328330993652,0.012841695785522462,0.01681350326538086]
text_encoder=[0.01319257640838623,0.01711609649658203,0.032640289306640625,0.047868961334228514,0.05910188674926758]
values5=t_t=[text_preprocess[i]+text_encoder[i] for i in range(len(text_preprocess))]
values6=text_decoder=[0.03734230422973633,0.04123081588745117,0.04280601501464844,0.043542976379394534,0.04510531234741211]

bs=[1,8,16,24,32]

bar_width = 1/7

r1 = np.arange(len(bs))
r2 = [x + bar_width for x in r1]
r3=[x+bar_width for x in r2]
r4=[x+bar_width for x in r3]
r5=[x+bar_width for x in r4]
r6=[x+bar_width for x in r5]

plt.bar(r1, values1, color='blue', width=bar_width, edgecolor='grey', label='image_open')
plt.bar(r2, values2, color='green', width=bar_width, edgecolor='grey', label='image_open+image_preprocess')
plt.bar(r3, values3, color='red', width=bar_width, edgecolor='grey', label='image_open+image_preprocess_visual_encoder')
plt.bar(r4, values4, color='orange', width=bar_width, edgecolor='grey', label='text_preprocess')
plt.bar(r5, values5, color='purple', width=bar_width, edgecolor='grey', label='text_preprocess+text_encoder')
plt.bar(r6, values6, color='cyan', width=bar_width, edgecolor='grey', label='text_decoder')

plt.xlabel('bs')
plt.ylabel('time')
plt.title('blip_vqa')
plt.xticks([r + 2.5 * bar_width for r in r1], bs)

plt.legend()

plt.savefig(root_path+"/blip_vqa_bar.png") 
