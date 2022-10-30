from run import *

ex_1 = MarketingcMain()
data = ex_1.prepro_data1("df_user_activities","df_users")

print(ex_1.check_revuenue_by_channel(data))

data2 = ex_1.prepro_data2("df_user_activities","df_users")
history,model = ex_1.model3(data2)
ex_1.check_model(history,model)