import cv2
import numpy as np
import os

folder = '/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg6/'
folder = '/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg29/'
# folder = '/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/figs/'

# # Read images : src image will be cloned into dst
# im = cv2.imread(os.path.join(folder, "bg_6.jpg"))
# obj = cv2.imread(os.path.join(folder, "doll.png"))
# mask = cv2.imread(os.path.join(folder, "doll_mask.png"))
# print(mask[:10, :10, ])
# mask = np.reshape(mask[:, :, 0], (mask.shape[0], mask.shape[1], 1))
# for i in range(mask.shape[0]):
#     for j in range(mask.shape[1]):
#         if mask[i, j, 0].item() != 0:
#             # print(i, j, mask[i, j, 0].item())
#             mask[i,j,0] = 255
# # Create an all white mask
# # mask = 255 * np.ones(obj.shape, obj.dtype)
#
# # The location of the center of the src in the dst
# width, height, channels = im.shape
# width_obj, height_obj, channels_obj = obj.shape
# center = (height // 2, width // 2)  # 融合的位置，可以自己设置
# print(center)
# # center = (140 + height_obj // 2, 160 + width_obj // 2)  # 融合的位置，可以自己设置
# center = (195 + height_obj // 2, 140 + width_obj // 2)  # 融合的位置，可以自己设置
# #
# print(center)
#
# print(im.shape, obj.shape, mask.shape)
# # Seamlessly clone src into dst and put the results in output
# # normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
# # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
# mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
# # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.)
# # mixed_clone_xx = cv2.
#
# # Write results
# # normal_clone 或者是 mixed_clone
# # cv2.imwrite(folder + "normal_merge.jpg", normal_clone)
# # cv2.imshow(mixed_clone)
# cv2.imwrite(folder + "fluid_merge_2.png", mixed_clone)


''' MSE comparison '''
# folder = '/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/figs/bg11/'
# folder = '/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/figs/bg13/'
# folder = '/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/figs/bg14/'
# gradient_res = "result_with_mask_not_qtree.png"
# gradient_res = "result_with_mask_not_qtree.png"
# # gradient_res = "result_with_mask_qtree.png"
# qtree_res = "result_with_mask_qtree.png"
# qtree_res = "result_oth_qtree.png"
# qtree_res = "result.png" # 0.09166208
# qtree_res = "result_with_mask_qtree_2.png" # 85.04398
# qtree_res = "result_with_mask_qtree_3.png" # 9.11631
# qtree_res = "result_with_mask_qtree_4.png" # 0.6570616
# # gradient_res = "result_with_mask_qtree_5.png" # 0.6615485
# qtree_res = "result_with_mask_qtree_6.png" # 0.65375173
# qtree_res = "result_with_mask_qtree_11.png" # 0.09166208
# qtree_res = "result_with_mask_qtree_12.png" # 0.4623 --- no x-1, y-1
# qtree_res = "result_with_mask_qtree_13.png" # 0.09085737 --- no x-1, y-1
# qtree_res = "result_with_mask_qtree_14.png" # 0.09072084 --- no x-1, y-1
# qtree_res = "result_with_mask_qtree_15.png" # 0.09046433 --- no x-1, y-1
# qtree_res = "result_with_mask_qtree_17.png" # 0.09046433 --- no x-1, y-1
# qtree_res = "result_with_mask_qtree_2.png" # 0.09046433 --- no x-1, y-1
# qtree_res = "result_with_mask_qtree.png" # 0.09046433 --- no x-1, y-1
#
# gradient_img = cv2.imread(os.path.join(folder, gradient_res))
# qtree_img = cv2.imread(os.path.join(folder, qtree_res))
# gradient_img = np.array(gradient_img, dtype=np.float32)
# qtree_img = np.array(qtree_img, dtype=np.float32)
#
# # mixed_img = cv2.imread(os.path.join(folder, "mixed.png"))
# # mixed_img = np.array(mixed_img)
# #
# # cvimg = cv2.imread(os.path.join(folder, "fluid_merge_2.png"))
# # cvimg = np.array(cvimg)
# #
# # delta_img = mixed_img - cvimg
# # cv2.imwrite(folder + "cv_delta.png", delta_img)
#
# print(gradient_img.shape, qtree_res)
#
# # mse = np.mean(np.sum((gradient_img - qtree_img) ** 2, axis=-1))
# mse = np.mean((gradient_img[:, :, 0] - qtree_img[:, :, 0]) ** 2)
# print(mse)


''' MSE calculation '''
# for i in range(1, 11):
#     qtree_img = os.path.join(folder, f"bg{i}", "result_with_mask_qtree.png")
#     print(qtree_img)
#     qtree_img = cv2.imread(qtree_img)
#     qtree_img = np.array(qtree_img)
#
#     gradient_img = os.path.join(folder, f"bg{i}", "result_with_mask_not_qtree.png")
#     print(gradient_img)
#     gradient_img = cv2.imread(gradient_img)
#     gradient_img = np.array(gradient_img)
#
#     print(qtree_img.shape, gradient_img.shape)
#
#     mse = np.mean((gradient_img[:, :, 0] - qtree_img[:, :, 0]) ** 2).item()
#     print(mse)
#
#     print(i, mse)

# tmp_comp = "/Users/meow/Downloads/xixik_2c2ebabc67f0babc.jpg"
# img = cv2.imread(tmp_comp)
#
# img = np.array(img)
#
# print(img.shape)
#
# # img = np.repeat(img, (5))
#
# img = np.concatenate([img for __ in range(5)], axis=0)
#
# cv2.imwrite("/Users/meow/Downloads/xixik_2c2ebabc67f0babc_comp.jpg", img)

# mask = np.ones_like(img)
#
# seam_finder = cv2.detail_GraphCutSeamFinder(cost_type="COST_COLOR_GRAD")
# seam_finder.find(img, (0, 0,), mask)
# print(type(mask))
# print(mask)

''' Basic composition '''
tmp_comp = "/Users/meow/Downloads/bg12_1.jpg"
tmp_comp = "/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg14/bg14_element.jpg"
tmp_comp = os.path.join(folder, "element.jpg")
img_1 = cv2.imread(tmp_comp)

img_1 = np.array(img_1)

# tmp_comp = "/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg14/bg14_element.jpg"
# img_2 = cv2.imread(tmp_comp)
#
# img_2 = np.array(img_2)
#
# print(img_1.shape, img_2.shape)

img_cbd = np.concatenate([img_1, img_1], axis=0)
# img_cbd = np.concatenate([img_1, img_2], axis=0)
# cv2.imwrite("/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg14/bg14_cbg.png", img_cbd)
cv2.imwrite(os.path.join(folder, "cbd.png"), img_cbd)

# a = [1500, 756]
# b = [600, 683]
# # 450, 36
# for aa, bb in zip(a,b):
#     print((aa - bb) // 2)
#
# src = "/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg13/bg13_1.jpg"
# src = cv2.imread(src)
#
# dst = "/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg13/bg13_2.jpg"
# dst = cv2.imread(dst)
#
# w, h, c = dst.shape
#
# mask = np.full((w, h, 1), fill_value=255, dtype=np.long)
# mask[0, :, :] = 0
# mask[:, 0, :] = 0
#
# print(src.shape, dst.shape, mask.shape)
#
# width, height, channels = src.shape
# # width_obj, height_obj, channels_obj = obj.shape
# center = (height // 2, width // 2)  # 融合的位置，可以自己设置
# # center = (0, 0)  # 融合的位置，可以自己设置
# # print(center)
# # # center = (140 + height_obj // 2, 160 + width_obj // 2)  # 融合的位置，可以自己设置
# # center = (195 + height_obj // 2, 140 + width_obj // 2)  # 融合的位置，可以自己设置
# # #
# # print(center)
# #
# # print(im.shape, obj.shape, mask.shape)
# # # Seamlessly clone src into dst and put the results in output
# # normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
# # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
# mixed_clone = cv2.seamlessClone(dst, src, mask, center, cv2.MIXED_CLONE)
# mixed_clone_pth = "/Users/meow/Study/_2021_autumn/Media Computing/Homeworks/HW1/image_composition/imgs/bg13/cvv.png"
# cv2.imwrite(mixed_clone_pth, mixed_clone)

# # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.)
# # mixed_clone_xx = cv2.
