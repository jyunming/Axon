import React from 'react';
import { usePaymentConfig } from '@hooks';

export const CheckoutBtn = () => {
  const paymentConfig = usePaymentConfig();
  return <button disabled={!paymentConfig.isRegionActive}>Checkout</button>;
};
