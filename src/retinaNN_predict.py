###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import pred_normal
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc_RGB


#========= CONFIG FILE TO READ FROM =======
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')

#original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
#the border masks provided by the DRIVE
DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(DRIVE_test_border_masks)
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
#====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')


# #ground truth
# gtruth= path_data + config.get('data paths', 'test_groundTruth')
# img_truth= load_hdf5(gtruth)
# visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'original')#.show()
# visualize(group_images(test_border_masks[0:20,:,:,:],5),'borders')#.show()
# visualize(group_images(img_truth[0:20,:,:,:],5),'gtruth')#.show()



#============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if average_mode == True:
    #patches_imgs_test, new_height, new_width, masks_test, patches_imgs_test_eval, patches_masks_test_eval = get_data_testing_overlap(
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
        Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
        patch_height = patch_height,
        patch_width = patch_width,
    )



#================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')
#Load the saved model
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print "predicted images size :"
print predictions.shape

#===== Convert the prediction arrays in corresponding images
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")



#========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
    original_imgs = np.copy(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
    orig_imgs = my_PreProc_RGB(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
    gtruth_masks = masks_test  #ground truth masks
else:
    pred_imgs = recompone(pred_patches,13,12)       # predictions
    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    original_imgs = orig_imgs
    gtruth_masks = recompone(patches_masks_test,13,12)  #masks
# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
#kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
## back to original dimensions
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print "Orig imgs shape: " +str(orig_imgs.shape)
print "pred imgs shape: " +str(pred_imgs.shape)
print "Gtruth imgs shape: " +str(gtruth_masks.shape)
#visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
visualize(group_images(original_imgs,N_visual),path_experiment+"all_originals")#.show()
visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions_soft")#.show()

pred_imgs_hard = np.copy(pred_imgs)
for i in range(pred_imgs_hard.shape[0]):
    pred_imgs_hard[i] = hard_pred(pred_imgs[i])

visualize(group_images(pred_imgs_hard,N_visual),path_experiment+"all_predictions_hard")#.show()
visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")#.show()
"""
#visualize results comparing mask and prediction:
assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(i))#.show()
"""


#====== Evaluate the results
print "\n\n========  Evaluate the results ======================="
#predictions only inside the FOV
y_scores, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks)  #returns data only inside the FOV
print "Calculating results only inside the FOV:"
print "y scores pixels: " +str(y_scores.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]) +" (584*565==329960)"
print "y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)"

#print "Evaluatation on whole images:"
#Macro average area under the precision-recall curve (PR_AUC)
macro_avg_PR_AUC = average_precision_score(y_true, y_scores, average='macro')
print "Macro average area under the precision-recall curve: " + str(macro_avg_PR_AUC)

micro_avg_PR_AUC = average_precision_score(y_true, y_scores, average='micro')
print "Micro average area under the precision-recall curve: " + str(micro_avg_PR_AUC)

"""
#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print "\nArea under the ROC curve: " +str(AUC_ROC)
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")
"""

"""
#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")
"""

#Confusion matrix
y_true_sparse = np.empty((y_true.shape[0]))
y_scores_sparse = np.empty((y_scores.shape[0]))

y_true_artery = np.copy(y_true_sparse)
y_scores_artery = np.copy(y_scores_sparse)

y_true_background = np.copy(y_true_sparse)
y_scores_background = np.copy(y_scores_sparse)

y_true_vein = np.copy(y_true_sparse)
y_scores_vein = np.copy(y_scores_sparse)

for i in range(y_true.shape[0]):
    y_true_sparse[i] = np.argmax(y_true[i])

    maximums = np.argwhere(y_scores[i] == np.amax(y_scores[i])) 
    if maximums.shape[0] > 1:
        y_scores_sparse[i] = 1
    else:
        y_scores_sparse[i] = maximums[0,0]

    y_true_artery[i] = y_true[i][0]
    y_true_background[i] = y_true[i][1]
    y_true_vein[i] = y_true[i][2]

    y_scores_artery[i] = y_scores[i][0]
    y_scores_background[i] = y_scores[i][1]
    y_scores_vein[i] = y_scores[i][2]

precision_artery, recall_artery, thresholds_artery = precision_recall_curve(y_true_artery, y_scores_artery)
precision_background, recall_background, thresholds_background = precision_recall_curve(y_true_background, y_scores_background)
precision_vein, recall_vein, thresholds_vein = precision_recall_curve(y_true_vein, y_scores_vein)

precision_artery, precision_background, precision_vein = np.fliplr([precision_artery])[0], np.fliplr([precision_background])[0], np.fliplr([precision_vein])[0]
recall_artery, recall_background, recall_vein = np.fliplr([recall_artery])[0], np.fliplr([recall_background])[0], np.fliplr([recall_vein])[0]

PRAUC_artery = np.trapz(precision_artery, recall_artery)
PRAUC_background = np.trapz(precision_background, recall_background)
PRAUC_vein = np.trapz(precision_vein, recall_vein)
print "Individual area under precision-recall curve for artery, background and vein:"
print PRAUC_artery, PRAUC_background, PRAUC_vein

plt.plot(recall_artery,precision_artery, 'r-', label='Area Under the Curve for Artery (AUC = %0.4f)' % PRAUC_artery)
plt.plot(recall_background,precision_background, 'g-', label='Area Under the Curve for Background (AUC = %0.4f)' % PRAUC_background)
plt.plot(recall_vein,precision_vein, 'b-', label='Area Under the Curve for Vein (AUC = %0.4f)' % PRAUC_vein)

plt.title('Precision-Recall curve to multi-class')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.savefig(path_experiment+"Precision_recall_multiclass.png")

confusion = confusion_matrix(y_true_sparse, y_scores_sparse, labels=[0, 1, 2])
print "Confusion matrix:"
print confusion

#Accuracy
correct = 0.0
for i in range(confusion.shape[0]):
    correct += confusion[i,i]
accuracy = correct / np.sum(confusion)
print "Global Accuracy: " +str(accuracy)

"""
#Confusion matrix
threshold_confusion = 0.5
print "\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion)
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print confusion
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print "Global Accuracy: " +str(accuracy)
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print "Specificity: " +str(specificity)
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print "Sensitivity: " +str(sensitivity)
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print "Precision: " +str(precision)
"""

"""
#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print "\nJaccard similarity score: " +str(jaccard_index)
"""

"""
#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print "\nF1 score (F-measure): " +str(F1_score)
"""

"""
print "\nEvaluation on patches:"
print patches_imgs_test.shape
print patches_imgs_test_eval.shape
y_scores_patches, y_true_patches = pred_normal(patches_imgs_test_eval, patches_masks_test_eval)

macro_avg_PR_AUC_patches = average_precision_score(y_true_patches, y_scores_patches, average='macro')
print "Macro average area under the precision-recall curve: " + str(macro_avg_PR_AUC_patches)

#Confusion matrix
y_true_sparse_patches = np.empty((y_true_patches.shape[0]))
y_scores_sparse_patches = np.empty((y_scores_patches.shape[0]))
for i in range(y_true_patches.shape[0]):
    y_true_sparse_patches[i] = np.argmax(y_true_patches[i])

    maximums = np.argwhere(y_scores_patches[i] == np.amax(y_scores_patches[i])) 
    if maximums.shape[0] > 1:
        y_scores_sparse_patches[i] = 1
    else:
        y_scores_sparse_patches[i] = maximums[0,0]

confusion_patches = confusion_matrix(y_true_sparse_patches, y_scores_sparse_patches, labels=[0, 1, 2])
print "Confusion matrix:"
print confusion_patches

#Accuracy
correct = 0.0
for i in range(confusion_patches.shape[0]):
    correct += confusion_patches[i,i]
accuracy_patches = correct / np.sum(confusion_patches)
print "Global Accuracy: " +str(accuracy_patches)

#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Whole image0:\n"
                +"Macro average area under the precision-recall curve: "+str(macro_avg_PR_AUC)
                +"\nConfusion matrix:\n"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\n\nPatches\n"
                +"Macro average area under the precision-recall curve: "+str(macro_avg_PR_AUC_patches)
                +"\nConfusion matrix:\n"
                +str(confusion_patches)
                +"\nACCURACY: " +str(accuracy_patches)
                )
file_perf.close()
"""

#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Macro average area under the precision-recall curves: "+str(macro_avg_PR_AUC)
                +"\n\nMicro average area under the precision-recall curves: "+str(micro_avg_PR_AUC)
                +"\n\nArea under precision-recall curve for artery class: "+str(PRAUC_artery)
                +"\n\nArea under precision-recall curve for background class: "+str(PRAUC_background)
                +"\n\nArea under precision-recall curve for vein class: "+str(PRAUC_vein)
                +"\n\nConfusion matrix:\n"
                +str(confusion)
                +"\n\nACCURACY: " +str(accuracy)
                )
file_perf.close()
