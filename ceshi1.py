# https://github.com/DIYer22/retail_product_checkout_tools
# https://rpc-dataset.github.io/#2-paper

# https://pytorch.org/docs/stable/search.html?q=torchvision&check_keywords=yes&area=default#
# https://pytorch.org/vision/stable/models.html
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=initialize%20model
# https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html
# https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py



import torchvision.models as models

# mobilenet_v2 = models.mobilenet_v2(pretrained=True)
# print('mobilenet_v2: ', mobilenet_v2)
# # print('mobilenet_v2.summary(): ', mobilenet_v2.summary())

# resnet152 = models.resnet152(pretrained=True)
# print('resnet152: ', resnet152)

alexnet = models.alexnet(pretrained=True)
print('alexnet: ', alexnet)
