
export default async ({ req, res, log, error }) => {
  if(req.method == 'GET'){

    return res.send("welcome to era of AI !")
  }
  if(req.method == 'POST'){

    return res.json({
      'sendData':req.body,
      // 'googleAPiKey': process.env.GOOGLE_GEMINI_API_KEY
    })
  }

};
